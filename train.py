# =============================================================================
# training/train.py — Training loop with gradient accumulation, LR scheduling,
# early stopping, and per-epoch checkpointing.
# =============================================================================

import time
import logging
from pathlib import Path

import torch
import torch.nn as nn

log = logging.getLogger(__name__)


class FocalLoss(nn.Module):
    """Multiclass focal loss (Lin et al., 2017).

    Plain cross-entropy treats every example equally, so the huge number of EASY,
    frequent-class examples dominates the gradient and the rare classes never get
    learned (here: E7/E27/E28 sit at F1 = 0.00). Focal loss multiplies each
    example's loss by (1 - p_true)^gamma: when the model is already confident
    (p_true → 1) the factor → 0, so easy examples are down-weighted and the model
    is forced to spend capacity on the hard, rare ones.

        loss = - alpha[class] * (1 - p_true)^gamma * log(p_true)

    `alpha` reuses the existing sqrt-inverse class weights (so rare classes keep an
    extra boost); `gamma` (default 2) controls how aggressively easy examples are
    suppressed. gamma = 0 reduces this to ordinary weighted cross-entropy.
    """

    def __init__(self, gamma: float = 2.0, weight=None):
        super().__init__()
        self.gamma = gamma
        self.weight = weight   # per-class alpha (a tensor) or None

    def forward(self, logits, target):
        log_probs = torch.nn.functional.log_softmax(logits, dim=1)
        logp_true = log_probs.gather(1, target.unsqueeze(1)).squeeze(1)  # log p_true
        p_true    = logp_true.exp()
        focal     = (1.0 - p_true) ** self.gamma * (-logp_true)
        if self.weight is not None:
            focal = focal * self.weight.gather(0, target)
        return focal.mean()


def _run_val_epoch(model, val_loader, device):
    """Runs one validation pass. Returns (avg_loss, accuracy).

    Uses an unweighted, unsmoothed CrossEntropyLoss internally.
    WHY: if the training criterion uses class weights and the model collapses
    to always predict a rare class, weighted val_loss stays artificially low
    (common classes have tiny weights) and early stopping never triggers.
    Unweighted val_loss reflects actual prediction quality.
    """
    val_criterion = nn.CrossEntropyLoss()
    model.eval()
    total_loss = 0.0
    correct    = 0
    total      = 0
    with torch.no_grad():
        for X_val, y_val in val_loader:
            X_val = X_val.to(device, non_blocking=True)
            y_val = y_val.to(device, non_blocking=True)
            logits      = model(X_val)
            total_loss += val_criterion(logits, y_val).item()
            correct    += int((logits.argmax(1) == y_val).sum())
            total      += len(y_val)
    avg_loss = total_loss / max(len(val_loader), 1)
    accuracy = correct / max(total, 1)
    return avg_loss, accuracy


def train_model(model, train_loader, val_loader, num_epochs, learning_rate,
                weight_decay, accum_steps, lr_patience, early_stopping_patience,
                checkpoint_dir, device,
                class_weights=None, label_smoothing=0.0,
                loss_type="weighted_ce", focal_gamma=2.0):
    """
    Trains the GRU with validation-based early stopping and LR scheduling.

    Validation split:
        A held-out val set is evaluated every epoch.  The LR scheduler and
        early stopping both monitor val_loss, not train_loss.  Using
        train_loss for scheduling caused the LR to never reduce because the
        model memorised training data quickly.

    Early stopping:
        If val_loss does not improve for `early_stopping_patience` epochs,
        training terminates.  This avoids overfitting and ensures the returned
        model is the best generalising one.

    Weight decay:
        L2 regularisation via Adam's weight_decay parameter further reduces
        overfit on rare-event classes.

    Args:
        model                    : GRUAnomalyDetector.
        train_loader             : Training DataLoader.
        val_loader               : Validation DataLoader.
        num_epochs               : Maximum epochs (early stopping may terminate sooner).
        learning_rate            : Initial Adam LR.
        weight_decay             : L2 regularisation coefficient.
        accum_steps              : Gradient accumulation steps.
        lr_patience              : Val-loss plateau epochs before LR halves.
        early_stopping_patience  : Val-loss plateau epochs before stopping.
        checkpoint_dir           : Directory for checkpoints.
        device                   : Compute device.
        class_weights            : Optional per-class loss weights.
        label_smoothing          : Label smoothing coefficient (0 = disabled).

    Returns:
        model: Best-val-loss model weights loaded in place.
    """
    ckpt_dir  = Path(checkpoint_dir)
    ckpt_dir.mkdir(exist_ok=True)
    best_ckpt = str(ckpt_dir / "best_model.pt")

    # Select the training criterion. Validation loss stays unweighted CE regardless
    # (see _run_val_epoch) so early stopping reflects real prediction quality.
    if loss_type == "focal":
        criterion = FocalLoss(gamma=focal_gamma, weight=class_weights)
        log.info("Loss: focal (gamma=%.1f, alpha=class_weights)", focal_gamma)
    else:
        criterion = nn.CrossEntropyLoss(
            weight          = class_weights,
            label_smoothing = label_smoothing,
        )
        log.info("Loss: weighted cross-entropy")
    # Weight decay in Adam applies L2 directly on parameters, complementing
    # label smoothing by penalising large weights that make predictions brittle.
    optimizer = torch.optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    # Schedule on val_loss (not train_loss) so LR reacts to generalisation stall.
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=lr_patience, min_lr=1e-6
    )

    log.info("Training for up to %d epochs (early stop patience=%d) on %s",
             num_epochs, early_stopping_patience, device)
    log.info("Effective batch = %d × %d = %d",
             train_loader.batch_size, accum_steps,
             train_loader.batch_size * accum_steps)

    best_val_loss  = float("inf")
    best_epoch     = 1
    no_improve     = 0

    for epoch in range(1, num_epochs + 1):
        # ---- Training ----
        model.train()
        total_loss  = 0.0
        num_batches = 0
        t0          = time.time()
        optimizer.zero_grad()

        for step, (X_batch, y_batch) in enumerate(train_loader, 1):
            X_batch = X_batch.to(device, non_blocking=True)
            y_batch = y_batch.to(device, non_blocking=True)
            logits  = model(X_batch)
            loss    = criterion(logits, y_batch) / accum_steps
            loss.backward()
            total_loss  += loss.item() * accum_steps
            num_batches += 1

            if step % accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()

        # Flush any remaining accumulated gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        optimizer.zero_grad()

        avg_train_loss = total_loss / max(num_batches, 1)

        # ---- Validation (unweighted loss — see _run_val_epoch docstring) ----
        avg_val_loss, val_acc = _run_val_epoch(model, val_loader, device)

        elapsed = time.time() - t0
        log.info(
            "Epoch [%3d/%d]  train_loss=%.4f  val_loss=%.4f  val_acc=%.2f%%"
            "  LR=%.2e  time=%.1fs",
            epoch, num_epochs, avg_train_loss, avg_val_loss,
            val_acc * 100, optimizer.param_groups[0]["lr"], elapsed,
        )

        # Schedule on val_loss so LR reacts to overfitting, not training progress
        scheduler.step(avg_val_loss)

        # ---- Checkpoint ----
        torch.save({
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "train_loss": avg_train_loss,
            "val_loss": avg_val_loss,
        }, str(ckpt_dir / f"epoch_{epoch:03d}.pt"))

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_epoch    = epoch
            no_improve    = 0
            torch.save(model.state_dict(), best_ckpt)
            log.info("  ✓ New best val_loss=%.4f saved", best_val_loss)
        else:
            no_improve += 1
            log.info("  No improvement %d/%d", no_improve, early_stopping_patience)

        # ---- Early stopping ----
        if no_improve >= early_stopping_patience:
            log.info("Early stopping triggered at epoch %d (best was epoch %d, val_loss=%.4f)",
                     epoch, best_epoch, best_val_loss)
            break

    model.load_state_dict(torch.load(best_ckpt, map_location=device))
    log.info("Training complete. Best val_loss=%.4f at epoch %d.",
             best_val_loss, best_epoch)
    return model
