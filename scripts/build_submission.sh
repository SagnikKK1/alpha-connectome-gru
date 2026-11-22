#!/usr/bin/env bash
# Pack trained members into a submission zip: an int8 C kernel (native/wnnfast.c) with an ONNX fallback.
#   scripts/build_submission.sh OUT_NAME runs/member_s804/ema_r40e9998_final.pt runs/member_s806/... [more]
# The builder refuses to package unless the fp32 graph matches the torch member mean and the int8 blob is within
# quantisation noise of it. Then the contract test replays the official row-by-row protocol on two valid sequences.
# Optional: PREBUILT=path/to/wnnfast_x86_64.so ships a precompiled kernel as the second fallback (the zip compiles
# wnnfast.c with gcc on the scorer first, then tries the prebuilt library, then falls back to ONNX).
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-python}
OUT=$1; shift
CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=4 $PY native/native_build.py "$@" \
  --out "out/$OUT" --fast-gelu --mults 0.35,0.35 --p0 a16w8 --onnx-members 0,1,2 ${PREBUILT:+--prebuilt "$PREBUILT"}
$PY native/contract_test.py "out/submission_$OUT.zip" 2
echo "built out/submission_$OUT.zip"
