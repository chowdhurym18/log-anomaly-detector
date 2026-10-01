# OpenStack — what I found across all four experiments

_This pulls together the four OpenStack experiments I ran: the sequence detector, the timing
follow-up, the parameter follow-up, and the control that checks the parameter result. 

## Short version

The GRU predicts the next event almost perfectly on OpenStack (99.1% Top-1), but it detects
almost nothing. The reason is the dataset. There are only 19 distinct event sequences across 2,067 sessions,
and 194 of the 197 anomalous sessions have an event sequence that also shows up in normal
sessions. So most anomalies look exactly like normal traffic to a model that only sees the
event order.

I tried adding timing. It didn't help. I tried adding the message parameters that template
parsing throws away. That looked better at first (AUROC 0.6643 vs 0.5205). But then I ran a
control using only normal sessions, and the same parameter features could tell
`openstack_normal1.log` apart from `openstack_normal2.log` at AUROC 0.7264 — higher than the
anomaly result, on data with no anomalies in it at all.

So I can't say the parameter model was detecting anomalies. It might just be picking up on
which log file a session came from.

## The setup

All four experiments use the same data and the same split.

- 2,067 VM sessions, one session per VM instance UUID
- 1,870 normal, 197 anomalous
- A session is anomalous if it came from `openstack_abnormal.log`.
- 198 sessions are in the abnormal file but one has a single log event, which is too short to
  make a context→target window. So 197 get evaluated.
- Stratified split, seed 42: train 1,446 / val 207 / test 414. The test set has 39 anomalies.
- The GRU only trains on normal sessions. The 138 anomalous training sessions get dropped
  before training.

The GRU, the windowing, the loss and the seed are the same ones used for HDFS and BGL. I did
not change the model for any of these experiments.

## Experiment 1 — the GRU on event sequences

### Next-event prediction is nearly perfect

| Metric | OpenStack | HDFS | BGL |
|---|---:|---:|---:|
| Top-1 accuracy | **99.1%** | 91.0% | 86.4% |
| Top-3 accuracy | 100.0% | 99.2% | 91.7% |
| Top-5 accuracy | 100.0% | 99.9% | 92.2% |
| Weighted F1 | 0.9908 | 0.9066 | 0.8598 |

This is the best next-event score of the three datasets. 

### Anomaly detection does not work

| Metric | Value |
|---|---:|
| F1 | 0.1722 |
| Precision | 0.0942 |
| Recall | 1.0000 |
| PR-AUC | 0.1185 |
| AUROC | 0.5205 |

Confusion matrix: TN 0, FP 375, FN 0, TP 39. The detector flagged all 414 test sessions.

Recall is 1.0 and precision equals the base rate, which means it flagged everything. The F1
of 0.1722 is exactly what you get for flagging everything at a 9.42% base rate. 

I checked all four block scores the project uses (`max_surprisal`, `mean_surprisal`,
`topk_miss_frac`, `nll_-logp_mean`). All of them land on the same flag-everything point, and
the best AUROC across all four is 0.5222. So this isn't a bad choice of score.

### Why good prediction and bad detection happen together

The detector works by being surprised. It learns what normal sequences look like, then flags
sessions where the next event was unexpected. If the model is never surprised, there is
nothing to flag.

OpenStack sessions are almost all the same. There are only **19 distinct event sequences
across 2,067 sessions**, and one sequence is 83.8% of all the traffic. The workload is a
scripted create/destroy cycle, so every VM does roughly the same thing. That makes the
next-event task easy and the detection task impossible.

**194 of the 197 anomalous sessions have an event-template sequence that is byte-identical
to a sequence that also appears in normal sessions.**

Those sessions and their normal twins are the same thing as far as the model can see. They
get the same score. No threshold can separate them, because there is nothing to separate. 

| Dataset | Anomalies sharing a normal sequence | Recall ceiling | Measured F1 |
|---|---:|---:|---:|
| HDFS | 0 of 1,797 | 100.0% | 0.674 |
| BGL | 9 of 2,000 | 99.6% | 0.917 |
| OpenStack | 194 of 197 | **1.5%** | 0.172 |


## Experiment 2 — adding timing

The sequence representation throws away timestamps, so I checked whether timing helps. I
built 8 session-level features (duration, event count, mean/median/std/min/max gap between
events, event rate) and ran three versions through the same logistic regression: GRU score
only, timing only, and both.

| Config | Precision | Recall | F1 | PR-AUC | AUROC | Flagged |
|---|---:|---:|---:|---:|---:|---:|
| GRU only | 0.0942 | 1.0000 | 0.1722 | 0.1185 | 0.5205 | 414 |
| Timing only | 0.1086 | 0.8462 | 0.1924 | 0.1081 | 0.5524 | 304 |
| GRU + timing | 0.1104 | 0.8718 | 0.1960 | 0.1221 | 0.5758 | 308 |

**Adding timing did not help.** F1 goes up from 0.1722 to 0.1960, which is not a real
improvement:

- None of the 8 timing features separate the classes on the training data. The smallest
  p-value was 0.094, which is not significant.
- Median session duration is 43.7 seconds for normal sessions and 43.7 seconds for anomalous
  ones. They are the same.

The F1 only moved because the classifier flagged fewer sessions (308 instead of 414) while
also missing more real anomalies. 

## Experiment 3 — adding message parameters

Template parsing replaces the variable parts of a message with `<*>`. So
`Took 19.05 seconds to spawn` becomes `Took <*> seconds to spawn`, and the GRU never sees
the 19.05. I wanted to know whether those discarded values carry the missing information.

The structured CSV doesn't keep the raw message, so I re-parsed the raw logs with the same
miner settings and the same file order, which gives back the same 40 templates. Then I
matched each message against its template to pull out the masked values. That worked on all
55,683 lines with an instance ID, with no failures.

That gave 50 parameter slots (a slot is one `<*>` position in one template):

- 23 identifiers — UUIDs, IP addresses, URL paths, long hex strings. **I excluded these.**
  If I kept them the model could just memorise which session is which instead of learning
  anything about behaviour.
- 20 numeric — API response times, response sizes, `Took <*> seconds` durations, memory/disk/
  vcpu amounts, instance counts.
- 7 categorical — lifecycle states like `Started`/`Paused`/`Resumed`/`Stopped`.

After dropping identifiers, constant slots and slots that show up in under 10% of training
sessions, 13 slots were left, which became 65 features.

| Config | Precision | Recall | F1 | PR-AUC | AUROC | Flagged |
|---|---:|---:|---:|---:|---:|---:|
| GRU only | 0.0942 | 1.0000 | 0.1722 | 0.1185 | 0.5205 | 414 |
| Parameters only | 0.1739 | 0.2051 | 0.1882 | 0.1596 | **0.6643** | 46 |
| GRU + parameters | 0.1667 | 0.2051 | 0.1839 | 0.1585 | 0.6623 | 48 |

This looked more promising than timing:

- AUROC went from 0.5205 to 0.6643.
- The bootstrap confidence interval on the improvement was [+0.0420, +0.2393], which does
  not include zero. So unlike timing, this was not just noise.
- At a fixed budget it ranked better: top 39 found 7 anomalies instead of 6, top 100 found 19
  instead of 11, top 200 found 27 instead of 18.
- On the 38 sequence-identical anomalies in the test set — the ones the sequence provably
  can't reach — AUROC went from 0.508 to 0.660.
- 9 of the 65 features separated the classes at BH q < 0.05, mostly API response times and
  operation durations.

It still wasn't a usable detector. F1 barely moved, precision was 0.167, and it only caught
8 of 39 anomalies at the operating threshold. The effect sizes were small too — the strongest
feature had medians of 0.2606 for normal and 0.2643 for anomalous, which is close. And the
directions didn't agree with each other: some timings were higher for anomalous sessions and
some were lower.

But the ranking improvement was real, so it needed checking properly.

## Experiment 4 — the control

Here is the problem with experiment 3. Under our ground truth, a session is anomalous if and
only if it came from `openstack_abnormal.log`. That file was recorded on a different day from
the two normal files. So "anomalous" and "which file it came from" are the same thing in this
data.

The features that carried the signal were API response times and operation durations. Those
are exactly the kind of numbers that change when a cluster is under different load. So the
model might be learning "this came from the abnormal capture" instead of "this VM failed".

To test that, I ran the same features and the same model on a question with no anomalies in
it at all:

> Can the parameter features tell `openstack_normal1.log` apart from `openstack_normal2.log`?

Both of those are normal. I excluded the abnormal file completely — no abnormal session was
used for features, training, thresholds or scoring. I also dropped the one session that shows
up in both normal files, since its source is ambiguous.

- normal1: 556 sessions (class 0)
- normal2: 1,313 sessions (class 1)
- Fresh stratified split on source, seed 42: train 1,308 / val 187 / test 374

**Result: AUROC 0.7264, 95% CI [0.6692, 0.7804].**

I also re-ran it using the exact same 65-feature slot set from experiment 3, and got the
identical 0.7264.

| Task | What it separates | AUROC |
|---|---|---:|
| GRU sequence baseline | anomalous vs normal | 0.5205 |
| Parameters (experiment 3) | anomalous vs normal | 0.6643 |
| **Control (experiment 4)** | **normal1 vs normal2, both normal** | **0.7264** |

The control scored **higher** than the anomaly task. The bottom of its confidence interval
(0.6692) is above the anomaly AUROC (0.6643). So on data with no anomalies in it, these
features are at least as good at telling groups apart as they were on the actual anomaly
task.

**This means we cannot be sure the parameter model was detecting anomalies.** It can clearly
tell captures apart, and in this dataset the anomaly label is the capture.

## What this all means

The main issue with OpenStack is not simply that the model is weak. The dataset has a problem
where anomaly status is tied to the file/capture the logs came from. Because of this, it is
difficult to tell whether the model is detecting failures or just differences between the log
collections.

There are really two separate problems stacked on top of each other:

1. **The event sequence can't see the anomalies.** 194 of 197 anomalous sessions have the
   same event sequence as normal sessions. That caps a sequence-only detector at about 1.5%
   recall no matter what model you use.
2. **Anything that can see a difference is confounded.** Because anomalous means "from the
   abnormal file", any feature that differs between the two captures will look like it
   works. The control shows this is not hypothetical — it actually happens with the parameter
   features.

For comparison, the same pipeline gets F1 0.674 on HDFS and 0.917 on BGL with no changes. So
the approach itself works. It just doesn't work here.
