# BGL dataset

BGL is an open log dataset collected from a BlueGene/L supercomputer at
Lawrence Livermore National Labs (LLNL), with 4,747,963 log messages of which
348,460 are labeled as alerts. It is used here as the second-dataset
validation for the suspicious-activity detector
(`main.py --dataset bgl --detect`).

Only the derived Drain template table (`BGL_templates.csv`) is committed in
this repository. The raw log and the derived per-node traces are large and are
**not** redistributed here — download and regenerate them:

1. Download `BGL.log` from LogHub
   (https://github.com/logpai/loghub — Zenodo record: https://zenodo.org/records/8196385)
   and place it at `data/BGL/BGL.log`.
2. Run `venv/bin/python bgl_preprocess.py` (requires `drain3`, see
   requirements.txt) to regenerate `BGL_structured.csv`, `Event_traces.csv`,
   and `anomaly_label.csv`.

### Citation

If you use the BGL dataset in your research, please cite:

+ Adam Oliner, Jon Stearley. [What Supercomputers Say: A Study of Five System
  Logs](https://ieeexplore.ieee.org/document/4273008), in Proc. of the 37th
  Annual IEEE/IFIP International Conference on Dependable Systems and Networks
  (DSN), 2007.
+ Jieming Zhu, Shilin He, Pinjia He, Jinyang Liu, Michael R. Lyu. [Loghub: A
  Large Collection of System Log Datasets for AI-driven Log
  Analytics](https://arxiv.org/abs/2008.06448). IEEE International Symposium
  on Software Reliability Engineering (ISSRE), 2023.
