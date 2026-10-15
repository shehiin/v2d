#!/bin/bash
# Train + evaluate every evaluation episode with tr.sh, objects from the kit roster (slot order).
# Skips episodes already evaluated; waits for an episode's bundles if they are not there yet.
#   trall.sh <root> <recon dir> <env steps> [episode ...]
R=$1 OUT=$2 STEPS=$3; shift 3
ROSTER=$R/v2d_submission_kit/data/track_3_evaluation_objects.json
eps=${*:-$(python3 -c "import json;print(' '.join(json.load(open('$ROSTER'))['episodes']))")}
for e in $eps; do
  objs=$(python3 -c "import json;print(' '.join(json.load(open('$ROSTER'))['episodes']['$e']))")
  res=$R/runs/e$e/res_$STEPS/episode_$(printf %06d $e).parquet
  [ -f $res ] && { echo "=== e$e done already"; continue; }
  for o in $objs; do
    until [ -f $R/$OUT/e${e}_$o/result_slam_gravity_aligned/result.npz ]; do sleep 120; done
  done
  SAVE=$STEPS $R/t3/tr.sh $R $OUT $e $STEPS $objs > $R/runs/tr_e$e.log 2>&1 \
    && echo "=== e$e ok $(date +%T)" || echo "=== e$e FAIL $(date +%T)"
done
echo "=== ALL TRAINED"
