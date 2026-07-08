# =============================================================================
# models/gru_model.py — GRU next-event prediction model (3 selectable variants).
#
# One class, three architectures selected by two flags:
#   • gru             : Embedding → LN → GRU            → LN → Dropout → Linear   (default)
#   • bigru           : Embedding → LN → BiGRU          → LN → Dropout → Linear
#   • bigru_attention : Embedding → LN → BiGRU → Attn   → LN → Dropout → Linear
#
# The DEFAULTS (bidirectional=False, use_attention=False) reproduce the original
# architecture exactly, so the existing .checkpoints/best_model.pt still loads.
#
# Key design decisions:
#   • LayerNorm after embedding: stabilises embedding gradient scale.
#   • LayerNorm after the recurrent summary: stabilises the hidden-state
#     distribution across sequences of varying event-frequency.
#   • Bidirectionality is SAFE here (no future leakage). The dataset predicts
#     event[i+seq_len] from the window event[i : i+seq_len]; the target sits
#     OUTSIDE the window, so reading the window in both directions never sees
#     the label. (Bidirectionality only leaks in per-timestep next-token setups,
#     which this is not — we make a single prediction one step past a fixed
#     window.)
# =============================================================================

import torch
import torch.nn as nn


class AttentionPooling(nn.Module):
    """Additive (Bahdanau-style) attention pooling over the time axis.

    A unidirectional/last-timestep model squeezes the whole window into one
    hidden vector and forgets which events mattered. Attention instead learns a
    relevance score for EACH timestep, turns the scores into weights that sum to
    1 (softmax), and returns the weighted average of the hidden states. The same
    weights are returned so we can SHOW which events the model focused on.

    Shapes:
        H        : (batch, time, hidden)   hidden state at every timestep
        scores   : (batch, time)           one scalar relevance score per step
        weights  : (batch, time)           softmax(scores) — sums to 1 over time
        context  : (batch, hidden)         Σ_t weights[t] · H[t]
    """

    def __init__(self, hidden_dim: int):
        super().__init__()
        # One learned linear layer maps each timestep's hidden vector to a single
        # relevance score. This is the entire attention mechanism — no heads, no
        # transformer block, ~hidden_dim parameters.
        self.score = nn.Linear(hidden_dim, 1)

    def forward(self, H):
        scores  = self.score(H).squeeze(-1)          # (B, T): drop the size-1 dim
        weights = torch.softmax(scores, dim=1)       # (B, T): non-negative, sum=1
        context = (weights.unsqueeze(-1) * H).sum(1) # (B, hidden): weighted average
        return context, weights


class GRUAnomalyDetector(nn.Module):
    """
    Sequence-next-event prediction model.

    Embedding → LayerNorm → (Bi)GRU → [Attention] → LayerNorm → Dropout → Linear

    Why this works for anomaly detection:
        Trained on normal logs, the model learns P(next_event | sequence). At
        inference, anomaly_score = f(1 − P(true_next | history), H(output),
        TopK_miss). A high score signals a sequence that deviates from learned
        normal patterns regardless of which class the model is biased toward.

    Args:
        bidirectional : if True, the GRU reads the window forwards AND backwards
                        (richer context; safe — the target is outside the window).
        use_attention : if True, pool every timestep with learned attention
                        instead of taking only the last timestep. Requires the
                        recurrent outputs over all timesteps.
    """

    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers, dropout,
                 bidirectional: bool = False, use_attention: bool = False,
                 pretrained_embeddings=None, freeze_embeddings: bool = False):
        super().__init__()
        self.bidirectional = bidirectional
        self.use_attention = use_attention

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=None)
        self.emb_norm  = nn.LayerNorm(embedding_dim)
        self.gru = nn.GRU(
            input_size    = embedding_dim,
            hidden_size   = hidden_dim,
            num_layers    = num_layers,
            batch_first   = True,
            dropout       = dropout if num_layers > 1 else 0.0,
            bidirectional = bidirectional,
        )
        # A bidirectional GRU concatenates the forward and backward hidden states,
        # so its per-timestep output is 2× wide. Everything downstream sizes to it.
        out_dim = hidden_dim * (2 if bidirectional else 1)

        self.attention  = AttentionPooling(out_dim) if use_attention else None
        self.hidden_norm = nn.LayerNorm(out_dim)
        self.dropout     = nn.Dropout(dropout)
        self.fc          = nn.Linear(out_dim, vocab_size)

        self._init_weights()

        # Optional semantic embedding initialisation (Phase 3). When a (vocab, emb_dim)
        # matrix is supplied, it REPLACES the random embedding init so similar templates
        # start close together. Default None → random init, so existing checkpoints and
        # the state_dict layout are completely unchanged.
        if pretrained_embeddings is not None:
            with torch.no_grad():
                self.embedding.weight.copy_(torch.as_tensor(
                    pretrained_embeddings, dtype=self.embedding.weight.dtype))
            if freeze_embeddings:
                self.embedding.weight.requires_grad_(False)

    def _init_weights(self):
        nn.init.uniform_(self.embedding.weight, -0.1, 0.1)
        nn.init.zeros_(self.fc.bias)
        nn.init.xavier_uniform_(self.fc.weight)

    def forward(self, x, return_attention: bool = False):
        # x: (batch, seq_len)
        emb        = self.emb_norm(self.embedding(x))   # (B, T, emb)
        gru_out, _ = self.gru(emb)                       # (B, T, out_dim)

        if self.attention is not None:
            # Attention pooling: a weighted average over ALL timesteps.
            pooled, weights = self.attention(gru_out)    # (B, out_dim), (B, T)
        else:
            # Original behaviour: summarise the window by its last timestep.
            pooled, weights = gru_out[:, -1, :], None    # (B, out_dim)

        h      = self.dropout(self.hidden_norm(pooled))  # (B, out_dim)
        logits = self.fc(h)                              # (B, vocab_size)

        if return_attention:
            return logits, weights
        return logits
