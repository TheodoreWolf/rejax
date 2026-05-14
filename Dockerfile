FROM nvidia/cuda:12.4.0-runtime-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    python3-pip \
    tmux \
    vim \
    git \
    curl \
    libglfw3-dev \
    libglfw3 \
    xvfb \
    libgl1-mesa-dev \
    libgl1-mesa-glx \
    libglew-dev \
    libosmesa6-dev \
    libglu1-mesa-dev \
    patchelf \
    xserver-xorg-dev \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install --upgrade pip setuptools wheel

ARG USERNAME
ARG USER_UID
ARG USER_GID

RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID --create-home $USERNAME \
    && chown -R $USER_UID:$USER_GID /home/$USERNAME

COPY . /home/$USERNAME/workspace
RUN chown -R $USER_UID:$USER_GID /home/$USERNAME/workspace

USER $USERNAME

WORKDIR /home/$USERNAME/workspace

# Install project deps first so they determine the base jax version,
# then install the CUDA-enabled jax on top so it provides GPU support
# without being downgraded by transitive deps.
ARG USE_CUDA=true
RUN pip install -e ".[compat,experiment]" && \
    if [ "$USE_CUDA" = true ]; then \
        pip install "jax[cuda12]==0.5.3" -f "https://storage.googleapis.com/jax-releases/jax_cuda_releases.html"; \
    fi

CMD ["/bin/bash"]