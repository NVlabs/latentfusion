#!/bin/bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

set -e
set -v

CONFIG_PATH="${DIR}/../resources/open3d_config.json"
UOIS_PYTHON=/home/kpar/.pyenv/versions/miniconda3-4.3.30/envs/uois/bin/python
UOIS_DATA=/home/kpar/data/uois
UOIS_SRC=/home/kpar/projects/uois
OPEN3D_DIR=/home/kpar/src/Open3D

echo "*** Copying Open3D configuration"
cp $CONFIG_PATH "$1/config.json"

if [[ ! -d "$1/mask" ]]; then
  echo "*** Generating masks"
  $UOIS_PYTHON "$UOIS_SRC/generate_realsense_masks.py" \
    --checkpoint-dir $UOIS_DATA \
    --data-dir $1 \
    --batch-size 12
#  python "$DIR/../tools/moped/generate_masks_plane.py" --data-dir "$1"
else
  echo "*** Masks already exist"
fi

#echo "*** Reconstructing"
#python "$OPEN3D_DIR/examples/Python/ReconstructionSystem/run_system.py" \
#  "$1/config.json" --make --register --refine --integrate
#
#echo "*** Processing scan"
#python "$DIR/../tools/moped/process_open3d_scan.py" "$1"
#
#echo "*** Generating plane masks"
#python "$DIR/../tools/moped/generate_masks_plane.py" --data-dir "$1" --grabcut
