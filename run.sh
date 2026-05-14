#!/bin/bash

# Usage: ./run_docker.sh [GPU_ID] [SWEEP_ID]
# If GPU_ID is omitted, uses all available GPUs
# If SWEEP_ID is provided, automatically starts wandb agent for that sweep

# Default to all GPUs if no argument provided
if [ -z "$1" ]; then
    echo "No GPU specified, using all available GPUs"
    GPU_ARG="--gpus all"
    GPU_NAME="all"
elif [ "$1" == "none" ]; then
    echo "No GPU specified, using CPU"
    GPU_ARG=""
    GPU_NAME="cpu"
else
    echo "Using GPU device(s): $1"
    GPU_ARG="--gpus '\"device=$1\"'"
    GPU_NAME=$(echo "$1" | tr ',' '-')
fi

USERNAME=$(whoami)

# Run the container interactively
eval docker run \
-it \
$GPU_ARG \
-v $(pwd):/home/$USERNAME/workspace \
-w /home/$USERNAME/workspace \
--shm-size=4g \
--name $(whoami)-rejax-$GPU_NAME \
--rm \
$(whoami)-rejax

echo "Container exited."
