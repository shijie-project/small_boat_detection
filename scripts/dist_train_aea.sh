#!/usr/bin/env bash
# Train the AEA recipe at a chosen model size. The config and the tuning (-t)
# checkpoint are derived from the size; the output dir is derived from the config
# name plus a fresh timestamp on every run, so repeated trainings never collide
# and the checkpoints + train.log live together under it.
#
#   bash scripts/dist_train_aea.sh [size]
#       size : s | m | l   (default m)
#
#   config : configs/dome/Dome-<SIZE>-AEA.yml      (override: CONFIG=...)
#   tuning : ckpts/Dome-<SIZE>-AITOD-best.pth      (override: TUNING=...)
#   outdir : output/Dome-<SIZE>-AEA/<timestamp>    (override: OUTDIR=...)
cd "$(dirname "$0")/.." || exit 1

# --- model size (positional arg, default m) ---------------------------------
MODEL_SIZE=${1:-${MODEL_SIZE:-m}}
MODEL_SIZE=$(echo "$MODEL_SIZE" | tr '[:upper:]' '[:lower:]')
case "$MODEL_SIZE" in
  s|m|l) ;;
  *) echo "ERROR: model size must be s, m, or l (got '$MODEL_SIZE')"; exit 1 ;;
esac
SIZE=$(echo "$MODEL_SIZE" | tr '[:lower:]' '[:upper:]')  # the S / M / L in the file names

export CUDA_VISIBLE_DEVICES=0,1

# --- resolve the Python interpreter -----------------------------------------
# Use ONLY the python in this venv folder; error out if it isn't there. No PATH
# / system / activated-venv fallbacks. Launch via `python -m torch.distributed.run`
# (not bare `torchrun`) so this exact interpreter is always the one used.
VENV=/data/shijili/venv/perceptia
PYTHON=""
for c in "${VENV}/bin/python" "${VENV}/bin/python3" "${VENV}/python" "${VENV}/Scripts/python.exe"; do
  [ -x "$c" ] && { PYTHON="$c"; break; }
done
[ -n "$PYTHON" ] || { echo "ERROR: no python found in ${VENV} (looked for bin/python, bin/python3, python, Scripts/python.exe)"; exit 1; }
echo "[py] using ${PYTHON} ($("$PYTHON" -c 'import sys;print("Python %d.%d.%d"%sys.version_info[:3])' 2>/dev/null))"

# derive config + tuning checkpoint from the model size (both overridable)
CONFIG=${CONFIG:-./configs/dome/Dome-${SIZE}-AEA.yml}
TUNING=${TUNING:-./ckpts/Dome-${SIZE}-AITOD-best.pth}
[ -f "$CONFIG" ] || { echo "ERROR: config not found: $CONFIG"; exit 1; }
[ -f "$TUNING" ] || { echo "ERROR: tuning checkpoint not found: $TUNING"; exit 1; }

# derive output dir: output/<config-basename>/<timestamp>
config_name=$(basename "${CONFIG}" .yml)
STAMP=$(date '+%Y%m%d-%H%M%S')
outdir=${OUTDIR:-output/${config_name}/${STAMP}}
mkdir -p "${outdir}"

echo "[$(date '+%F %T')] training ${config_name} -> ${outdir}"

"${PYTHON}" -m torch.distributed.run --master_port=7789 --nproc_per_node=2 train.py \
     -c "${CONFIG}" -t "${TUNING}" --seed=42 --output-dir "${outdir}" \
     2>&1 | tee -a "${outdir}/train.log"
