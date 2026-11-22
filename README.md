# Alpha Connectome: distilled GRU ensemble

My solution code for the [Wunder Fund Alpha Connectome](https://wundernn.io/connectome) RNN challenge (2026). The task
is to predict two future-return targets for a market instrument from anonymised order-book sequences. Predictions come one
row at a time, using only the past, on a CPU, inside a 60-minute limit.

Team Tick_Talk (with teammate CHUNG-YU-WEN) finished **57th** on the final private leaderboard. The build this repo
reproduces scored **0.668474** on the public leaderboard (weighted Pearson).

## The model in one paragraph

Each member is a single-layer GRU (hidden 256). In front of it sits an input projection, 112 features to 192, made of two
Linear/GELU/LayerNorm blocks. A small shared convolution runs over the 11 book levels and is folded into the first linear
layer at export. The head is an MLP that reads the GRU output together with the projected input. Training uses truncated
backprop in 500-row chunks, with the state carried through the whole 20,000-row sequence. The loss is half pooled
weighted Pearson and half a weighted MSE anchor. Each member is distilled from a bidirectional teacher that sees 100 rows
ahead. The optimiser is Muon on the hidden weight matrices and AdamW on everything else. The submission is the plain mean of
eight members' EMA weight copies, run through a hand-written int8 C kernel at about 8.6 µs per row per member.

## What actually moved the score

Measured at the ensemble level against a fixed control, in order of size:

| Change | Gain |
|---|---|
| Lookahead distillation from a bidirectional teacher (100-row window) | +0.003 |
| Muon optimiser instead of AdamW for the hidden matrices | +0.0018 (leaderboard) |
| 6 epochs and the book-level convolution | +0.0013 (screen) |
| Auxiliary head on the exact mid-price path (builder not included) | +0.0010 (screen) |
| Shipping EMA weight copies instead of raw weights | +0.00025 (leaderboard) |

Things that read flat or negative: other architectures (minGRU, sLSTM, SSMs, TCNs, attention pooling, transformers as
teachers), more capacity, every regulariser I tried, output transforms fitted on validation, routing by instrument family,
second-generation distillation, and other optimisers (SOAP, Lion, MARS, NorMuon). See [NOTES.md](NOTES.md) for the
measurement lessons, which mattered more than any single model change.

## Layout

```
src/train.py          the trainer (all flags; the recipe uses a small subset)
src/arch/refgru.py    the member architecture
src/losses.py         pooled weighted-Pearson and anchor losses
src/teacher_dump.py   dumps teacher predictions with a bounded lookahead
src/infer/            ONNX graph builders (the fallback path inside the zip)
native/               int8 C kernel, blob builder, solution.py template, protocol contract test
configs/recipes.json  exact argument lists for the teachers and members
configs/valid_split.json  fixed split of the validation set: dev 300 / holdout 500 / retrain 1,073
scripts/              cache builder, end-to-end training script, submission builder
```

## Running it

You need the competition data and starter pack from the organisers. They are not redistributed here.

1. Put `train.parquet` and `valid.parquet` in `datasets/`, and the starter pack's `utils.py` at the repo root.
2. `pip install -r requirements.txt` (PyTorch 2.x with CUDA for training; CPU is fine for building the zip).
3. `python scripts/build_caches.py` builds the memory-mapped caches (about 130 GB on disk).
4. `bash scripts/train_recipe.sh` trains the two teachers, dumps them, and trains the eight members.
5. Build the zip:
   `bash scripts/build_submission.sh my_build runs/member_s8{04,06,07,08,09,10,11,12}/ema_r40e9998_final.pt`

Every checkpoint is bit-reproducible from its arguments and seed: the trainer uses deterministic kernels by default
(`--nondeterministic` trades that for speed).

`train.py` carries every flag from two months of experiments. Most of them are closed research and stay off by default.
The comments that mention `offline/...` scripts or lettered research lanes refer to my working repo and are kept as history.

## License

MIT. See [LICENSE](LICENSE).
