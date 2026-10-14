#!/bin/bash
# One episode from reconstruction to a scored-format trajectory: task (multi.py) -> retarget to
# floating Sharpa hands -> rigid URDFs + support -> FlashCHORD -> kit evaluator.
#   tr.sh <root> <recon dir under root> <episode> <env steps> <object> [object ...]
# Body names get an e<episode>_ prefix: URDFs are written per body name into one shared folder.
# Training runs at motion_speed 1.0 on a 20 fps task, so evaluator step k is video frame k
# (the packer reads reference_frame as the frame index).
set -eo pipefail
R=$1 OUT=$2 E=$3 STEPS=$4; shift 4
S=e$E
names=(); bundles=()
for o in "$@"; do names+=("${S}_$o"); bundles+=("/w/$OUT/${S}_$o/result_slam_gravity_aligned"); done
U="--user $(id -u):$(id -g) -e HOME=/tmp"
RG="docker run --rm --gpus all $U --group-add 1234 -e ACCEPT_EULA=Y -e HUMAN_MOTION_DATA_DIR=/w/hmd
  -e ROBOTIC_GROUNDING_INTERMEDIATE_DIR=/w/tasks/inter -v $R/video_to_data:/workspace/video_to_data -v $R:/w
  -v $R/tasks/kit/data:/isaac-sim/kit/data -v $R/tasks/kit/cache:/isaac-sim/kit/cache
  -v $R/tasks/kit/logs:/isaac-sim/kit/logs -w /workspace/video_to_data/robotic_grounding
  --entrypoint bash robotic-grounding:latest -c"
FC="docker run --rm --gpus all $U -v $R:/w -v $R/runs/fclogs:/workspace/logs
  -v $R/runs/fccache:/workspace/src/flash_chord/assets/.cache --entrypoint bash flash_chord:latest -c"
P=/w/hmd/ego_recon/processed/sequence_id=$S/robot_name=sharpa_wave
cd $R; mkdir -p tasks/inter tasks/kit/{data,cache,logs} hmd/ego_recon/reconstructed_stage runs/fclogs runs/fccache
step() { echo "=== $S $1 $(date +%T)"; }

step task
docker run --rm --gpus all $U -e HUMAN_MOTION_DATA_DIR=/w/hmd -v $R:/w -w /w v2d_task_library_loader:latest \
  python /w/t3/multi.py --bundles "${bundles[@]}" --names "${names[@]}" --sequence_name $S --fps 20 \
  --mano_model_dir /w/mano --mesh_dir /w/hmd/ego_recon/processed \
  --object_model_root /w/hmd/ego_recon/processed --output_dir /w/tasks/loaded --no_ground_align --save
step retarget
$RG "python scripts/retarget/ego_recon_to_sharpa.py --input_dir /w/tasks/loaded --output_dir /w/hmd/ego_recon/processed --sequence_id $S --save"
step support
$RG "python scripts/generate_rigid_urdfs.py --dataset ego_recon && python scripts/reconstruct_support_surfaces.py --dataset ego_recon --input_dir /w/hmd/ego_recon/processed --sequence_id $S --robot_name sharpa_wave --output /w/hmd/ego_recon/reconstructed_stage/${S}_sharpa_wave_support.usda"
step train
SAVE=$((STEPS / 4))
$FC "python scripts/train_flash_sac.py experiment=sharpa_flash_sac \"task.parquet='$P'\" task.motion_speed=1.0 reset.seed=42 logging.mode=disabled output_dir=/w/runs/$S scene.world_count=512 scene.support=false training.total_environment_steps=$STEPS training.replay.capacity=524288 training.replay.minimum_size=65536 training.checkpoint.save_interval_environment_steps=$SAVE"
step eval
for ck in runs/$S/policy_*.safetensors; do
  n=$(basename $ck .safetensors); n=${n#policy_}
  echo "{\"$E\": \"/w/$ck\"}" > runs/$S/ck_$n.json
  $FC "python /w/v2d_submission_kit/examples/run_policy_evaluation.py --checkpoints /w/runs/$S/ck_$n.json --results /w/runs/$S/res_$n --runtime-root /workspace"
done
step done
