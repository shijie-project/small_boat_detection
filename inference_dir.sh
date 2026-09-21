#!/usr/bin/env bash
# Detect boats in a folder of tiles cut by tools/dataset/tile_satellite.py.
#
#   ./inference_dir.sh                       # pick a folder from a list
#   ./inference_dir.sh <tile folder>         # that folder
#   ./inference_dir.sh <split_images> --all  # every folder under it
#
# Results land in <tile folder>/inf_det/ (predictions.json in COCO results
# format, detections.json, detections.csv, tiles/*_det.jpg, <scene>_overlay.jpg).
export CUDA_VISIBLE_DEVICES=0

CONFIG=configs/dome/Dome-M-AEA.yml
CKPT=../ckpts/Dome-M-AEA-best.pth
COCO=../data/annotations/val_coco.json  # class names + the image_ids predictions.json reuses

INPUT=""
if [ -n "$1" ] && [ "${1#-}" = "$1" ]; then
  INPUT="-i $1"
  shift
fi

python tools/inference/torch_inf_dir.py \
    --config "$CONFIG" \
    --resume "$CKPT" \
    --coco "$COCO" \
    --device cuda:0 \
    --batch 8 \
    --thrh 0.4 \
    $INPUT "$@"
