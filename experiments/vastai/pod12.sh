#!/usr/bin/env bash
# POD 12 — takes over the LAST 9 runs of POD 8's queue
#   (HAM10000, full, noisy): ngc, murmura, engc x 3 seeds
#   -> 9 runs @ ~100 min => ~15h
# POD 8 does its remaining Part A (5 runs) + ham10000_full_noisy_cga_s{321,42,7}
#   and must be STOPPED after `ham10000_full_noisy_cga_s7` is DONE.
# Fully self-contained — no other file needed on the pod besides this one.
set -uo pipefail

REPO_DIR="${REPO_DIR:-$HOME/repo}"
DATA_DIR="${DATA_DIR:-$HOME/data}"
VENV_DIR="${VENV_DIR:-$HOME/ngc_env}"

# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"
cd "$REPO_DIR"

# Fix for gloo TCP hangs common on cloud/Docker pods (Vast AI included):
# without this, dist.new_group() can intermittently fail with
# "failed to recv, got 0 bytes" if gloo auto-picks a non-loopback interface.
export GLOO_SOCKET_IFNAME=lo

LOGDIR="$REPO_DIR/run_logs/pod12"
mkdir -p "$LOGDIR"

run_exp () {
    local dataset=$1 classes=$2 topology=$3 method=$4 epochs=$5 seed=$6 noise_profile=$7 tag=$8
    local logfile="$LOGDIR/${tag}.log"

    local extra_args=()
    if [ -n "$noise_profile" ]; then
        extra_args+=(--noise-profile "$noise_profile")
    fi

    local neighbor_args=()
    if [ "$topology" = "ring" ] || [ "$topology" = "chain" ]; then
        neighbor_args+=(--neighbors 2)
    elif [ "$topology" = "torus" ]; then
        neighbor_args+=(--neighbors 4)
    fi

    # Retry up to 3x: full/torus topologies (degree-4 graph, 20 sequential
    # dist.new_group() calls for world_size=5) have been observed to hang
    # intermittently on cloud pods with "failed to recv, got 0 bytes" —
    # this looks like container-network flakiness, not a deterministic bug,
    # so a retry is worth it before giving up.
    local attempt
    for attempt in 1 2 3; do
        echo "[$(date '+%F %T')] START $tag (attempt $attempt/3)"
        if python trainer.py \
            --data-dir "$DATA_DIR/$dataset" --dataset "$dataset" --classes "$classes" \
            --lr 0.01 --batch-size 160 --world_size 5 --skew 1 --gamma 0.1 \
            --normtype evonorm --arch cganet --momentum 0.9 --alpha 1.0 --nesterov \
            --graph "$topology" "${neighbor_args[@]}" \
            --optimizer "$method" --epochs "$epochs" --seed "$seed" \
            "${extra_args[@]}" \
            > "$logfile" 2>&1
        then
            echo "[$(date '+%F %T')] DONE   $tag (attempt $attempt)"
            return 0
        fi
        echo "[$(date '+%F %T')] FAILED $tag on attempt $attempt (see $logfile)"
    done
    echo "[$(date '+%F %T')] GAVE UP on $tag after 3 attempts"
    return 1
}


SEEDS=(321 42 7)
NOISE_ORIG="1:0.05,2:0.2,3:0.2,4:0.2"

for method in ngc murmura engc; do
    for seed in "${SEEDS[@]}"; do
        run_exp ham10000 7 full "$method" 100 "$seed" "$NOISE_ORIG" \
            "ham10000_full_noisy_${method}_s${seed}"
    done
done

echo "POD 12 finished."
