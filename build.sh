#!/bin/bash

echo "Building rejax container"
docker build -t $(whoami)-rejax -f Dockerfile . \
--build-arg USE_CUDA=true \
--build-arg USERNAME=$(whoami) \
--build-arg USER_UID=$(id -u) \
--build-arg USER_GID=998
# --no-cache