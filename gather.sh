#!/bin/bash
# collect evaluator outputs and reconstructed meshes in the packer's layout
#   gather.sh <root> <recon dir> <env steps> <out dir>
R=$1 OUT=$2 STEPS=$3 D=$4
ROSTER=$R/v2d_submission_kit/data/track_3_evaluation_objects.json
mkdir -p $D/results $D/meshes
for e in $(python3 -c "import json;print(' '.join(json.load(open('$ROSTER'))['episodes']))"); do
  E=$(printf %06d $e); res=$R/runs/e$e/res_$STEPS
  [ -f $res/episode_$E.parquet ] || { echo "missing rollout e$e"; continue; }
  cp $res/episode_$E.parquet $res/episode_$E.json $D/results/
  $R/.venv/bin/python $R/t3/names.py $e $D/results/episode_$E.parquet
  mkdir -p $D/meshes/episode_$E
  for o in $(python3 -c "import json;print(' '.join(json.load(open('$ROSTER'))['episodes']['$e']))"); do
    cp $R/$OUT/e${e}_$o/result_slam_gravity_aligned/mesh.obj $D/meshes/episode_$E/$o.obj
  done
done
ls $D/results | wc -l
