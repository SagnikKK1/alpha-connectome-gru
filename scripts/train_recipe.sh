#!/usr/bin/env bash
# The full recipe behind the final submission, end to end. One GPU is enough; each step is resumable by
# running it again with the finished parts commented out. Times are for one RTX-class 48 GB GPU.
#
#   1. two bidirectional lookahead teachers               ~70 min each
#   2. dump their predictions with a 100-row lookahead     ~1 GB on disk
#   3. eight causal student members distilled from them   ~3 h each
#   4. pack the members' EMA copies into a native zip      see scripts/build_submission.sh
#
# Every run is bit-reproducible from its arguments and seed (deterministic kernels are enforced).
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-python}
args() {  # recipe arguments; drops the exact-mid aux flags when their dumps are absent
  $PY - "$1" <<'PYEOF'
import json, os, sys
a = json.load(open("configs/recipes.json"))[sys.argv[1]]
if "--dens-mid" in a and not os.path.isdir(a[a.index("--dens-mid") + 1]):
    for f in ("--densaux", "--dens-mid"):
        i = a.index(f); del a[i:i + 2]
print(" ".join(a))
PYEOF
}

# 1. teachers (non-causal: they see the whole sequence; used only as training targets)
$PY src/train.py $(args BT_plain_s801_teacher) --tag BT_plain_s801
$PY src/train.py $(args BT_dir10_s801_teacher) --tag BT_dir10_s801

# 2. teacher dump: the mean of the two teachers, lookahead bounded to 100 rows
$PY src/teacher_dump.py --tags BT_plain_s801,BT_dir10_s801 --raw --window 100 \
    --caches train_full,valid_retrain,valid_all --out runs/teacher_bidir_w100

# 3. members. The shipped members train on all valid sequences and use seeds 804, 806..812.
#    --ema-extra saves extra weight averages beside the raw weights; r40e9998 is the copy that shipped.
EMA='e9997:0.9997:0,r40e9998:0.999:0:0.9998:0.6'
for s in 804 806 807 808 809 810 811 812; do
  $PY src/train.py $(args member_full | sed "s/--seed 804/--seed $s/") --ema-extra "$EMA" --tag member_s$s
done
