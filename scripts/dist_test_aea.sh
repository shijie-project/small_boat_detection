#!/usr/bin/env bash
# Evaluate (test-only) the AEA recipe at a chosen model size. The config is
# derived from the size; the checkpoint to evaluate is given via CKPT (-r). The
# eval.log is written next to the checkpoint so the run and its log live together.
#
#   bash scripts/dist_test_aea.sh [size]
#       size : s | m | l   (default s)
#
#   config   : configs/dome/Dome-M-AEA.yml          (override: CONFIG=...)
#   checkpoint (-r) : ckpts/Dome-<size>-AEA-best.pth (override: CKPT=...)
cd "$(dirname "$0")/.." || exit 1

# --- model size (positional arg, default s) ---------------------------------
MODEL_SIZE=${1:-${MODEL_SIZE:-s}}
MODEL_SIZE=$(echo "$MODEL_SIZE" | tr '[:upper:]' '[:lower:]')
case "$MODEL_SIZE" in
  s|m|l) ;;
  *) echo "ERROR: model size must be s, m, or l (got '$MODEL_SIZE')"; exit 1 ;;
esac

export CUDA_VISIBLE_DEVICES=0

# --- resolve the Python interpreter -----------------------------------------
# Use ONLY the python in this venv folder; error out if it isn't there. No PATH
# / system / activated-venv fallbacks. Run train.py directly with this exact
# interpreter (single-process, no torchrun / distributed launch).
VENV="C:/Shijie_Li/.venv"
PYTHON=""
for c in "${VENV}/bin/python" "${VENV}/bin/python3" "${VENV}/python" "${VENV}/Scripts/python.exe"; do
  [ -x "$c" ] && { PYTHON="$c"; break; }
done
[ -n "$PYTHON" ] || { echo "ERROR: no python found in ${VENV} (looked for bin/python, bin/python3, python, Scripts/python.exe)"; exit 1; }
echo "[py] using ${PYTHON} ($("$PYTHON" -c 'import sys;print("Python %d.%d.%d"%sys.version_info[:3])' 2>/dev/null))"

# derive config + checkpoint from the model size (both overridable)
CONFIG=${CONFIG:-./configs/dome/Dome-M-AEA.yml}
CKPT=${CKPT:-./ckpts/Dome-M-AEA-best.pth}
[ -f "$CONFIG" ] || { echo "ERROR: config not found: $CONFIG"; exit 1; }
[ -f "$CKPT" ]   || { echo "ERROR: checkpoint not found: $CKPT"; exit 1; }

# eval.log lives next to the checkpoint (train.py drops the run output there too)
config_name=$(basename "${CONFIG}" .yml)
logdir=$(dirname "${CKPT}")
mkdir -p "${logdir}"

echo "[$(date '+%F %T')] testing ${config_name} with ${CKPT} -> ${logdir}/eval.log"

"${PYTHON}" train.py \
     -c "${CONFIG}" -r "${CKPT}" --test-only --seed=42 \
     2>&1 | tee -a "${logdir}/eval.log"
