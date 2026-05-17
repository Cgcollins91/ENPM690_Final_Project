# syntax=docker/dockerfile:1
#
# ENPM690 Final Project Isaac Sim / Isaac Lab training image.
#
# Installs the runtime needed for the standalone training pipeline:
#   - Isaac Sim 5.1.0 and Isaac Lab
#   - installs this repo, its topdown IsaacLab tasks, and FastTD3 dependency
#   - defaults to a headless training-friendly shell
#
# Build:
#   docker build -t enpm690-final-project:latest .
#
# Headless dry run:
#   docker run --gpus all --rm -it \
#     --user "$(id -u):$(id -g)" \
#     -e ENPM690_PYTHON=/opt/conda/envs/unitree_sim_env/bin/python \
#     -v "$PWD/runs:/workspace/project/runs" \
#     -v "$PWD/runs_training:/workspace/project/runs_training" \
#     -v "$PWD/checkpoints:/workspace/project/checkpoints" \
#     enpm690-final-project:latest \
#     python scripts/launch_pipeline.py --run-dir runs_training/pipeline_fullreward_adaptive_bc_r1 --dry-run
#
# Headless training/eval:
#   docker run --gpus all --rm -it --network host \
#     --user "$(id -u):$(id -g)" \
#     -e ENPM690_PYTHON=/opt/conda/envs/unitree_sim_env/bin/python \
#     -v "$PWD/runs:/workspace/project/runs" \
#     -v "$PWD/runs_training:/workspace/project/runs_training" \
#     -v "$PWD/checkpoints:/workspace/project/checkpoints" \
#     enpm690-final-project:latest \
#     python scripts/launch_pipeline.py --run-dir runs_training/pipeline_fullreward_adaptive_bc_r1
#
# Direct modular trainer smoke:
#   docker run --gpus all --rm -it --network host \
#     --user "$(id -u):$(id -g)" \
#     -v "$PWD/runs:/workspace/project/runs" \
#     -v "$PWD/runs_training:/workspace/project/runs_training" \
#     enpm690-final-project:latest \
#     python -m training --task Isaac-Topdown-Curriculum-G129-Dex3-Joint \
#       --num-envs 1 --total-steps 1 --start-steps 0 --headless \
#       --eval-steps 0 --eval-episodes 0 --checkpoint-every 0 \
#       --rolling-checkpoint-every 0 --tensorboard-dir off
#
# GUI/X11 smoke:
#   xhost +local:docker
#   docker run --gpus all --rm -it --network host \
#     --user "$(id -u):$(id -g)" \
#     -e DISPLAY="$DISPLAY" \
#     -e NVIDIA_VISIBLE_DEVICES=all \
#     -e NVIDIA_DRIVER_CAPABILITIES=compute,utility,video,graphics,display \
#     -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
#     -v /etc/vulkan/icd.d:/etc/vulkan/icd.d:ro \
#     -v /usr/share/vulkan/icd.d:/usr/share/vulkan/icd.d:ro \
#     -v "$PWD/runs:/workspace/project/runs" \
#     -v "$PWD/runs_training:/workspace/project/runs_training" \
#     enpm690-final-project:latest bash

FROM nvidia/cuda:12.2.0-runtime-ubuntu22.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC
ENV CONDA_DIR=/opt/conda
ENV PATH=${CONDA_DIR}/bin:${PATH}
ENV OMNI_KIT_ACCEPT_EULA=YES
ENV ACCEPT_EULA=Y
ENV PRIVACY_CONSENT=Y
ENV NVIDIA_DRIVER_CAPABILITIES=all

ARG MINICONDA_URL=https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
ARG ISAAC_SIM_VERSION=5.1.0
ARG ISAACLAB_COMMIT=80094be3245aa5c8376a7464d29cb4412ea518f5

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ca-certificates \
    cmake \
    curl \
    git \
    git-lfs \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libglu1-mesa-dev \
    libxt6 \
    libvulkan1 \
    vulkan-tools \
    wget \
    && rm -rf /var/lib/apt/lists/*

RUN wget -q "${MINICONDA_URL}" -O /tmp/miniconda.sh && \
    bash /tmp/miniconda.sh -b -p "${CONDA_DIR}" && \
    rm /tmp/miniconda.sh && \
    "${CONDA_DIR}/bin/conda" clean -afy

RUN conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main && \
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r && \
    conda create -n unitree_sim_env python=3.11 -y && \
    conda clean -afy

SHELL ["conda", "run", "-n", "unitree_sim_env", "/bin/bash", "-c"]

RUN conda install -y -c conda-forge "libgcc-ng>=12" "libstdcxx-ng>=12" && \
    conda clean -afy

RUN pip install --upgrade pip && \
    pip install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 \
        --index-url https://download.pytorch.org/whl/cu128 && \
    pip install "isaacsim[all,extscache]==${ISAAC_SIM_VERSION}" \
        --extra-index-url https://pypi.nvidia.com && \
    pip install numpy==1.26.0 opencv-python-headless==4.11.0.86

WORKDIR /workspace

RUN git clone https://github.com/isaac-sim/IsaacLab.git IsaacLab && \
    cd IsaacLab && \
    git checkout "${ISAACLAB_COMMIT}" && \
    ./isaaclab.sh --install

WORKDIR /workspace/project

COPY pyproject.toml requirements.txt ./
RUN pip install -r requirements.txt

COPY README.md isaacsim_compat.py run.py ./
COPY LICENSE NOTICE ./
COPY assets/ ./assets/
COPY robots/ ./robots/
COPY scripts/ ./scripts/
COPY src/ ./src/
COPY tasks/ ./tasks/
COPY training/ ./training/

ENV PYTHONPATH=/workspace/project/src:/workspace/project

RUN python -m py_compile \
        run.py \
        isaacsim_compat.py \
        scripts/launch_pipeline.py \
        scripts/eval_checkpoint.py \
        scripts/eval_visualization.py && \
    python -m compileall -q training src tasks


FROM nvidia/cuda:12.2.0-runtime-ubuntu22.04 AS runtime

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC
ENV CONDA_DIR=/opt/conda
ENV PATH=${CONDA_DIR}/bin:${PATH}
ENV OMNI_KIT_ACCEPT_EULA=YES
ENV ACCEPT_EULA=Y
ENV PRIVACY_CONSENT=Y
ENV NVIDIA_DRIVER_CAPABILITIES=all
ENV OMNI_KIT_ALLOW_ROOT=1
ENV OMNI_KIT_DISABLE_STARTUP=1
ENV ISAACLAB_PATH=/workspace/IsaacLab
ENV PROJECT_ROOT=/workspace/project
ENV PYTHONPATH=/workspace/project/src:/workspace/project
ENV PYTHONUNBUFFERED=1
ENV UNITREE_TASKS_IMPORT_FILTER=tasks.g1_tasks.cgc_topdown_curriculum_g1_29dof_dex3
ENV UNITREE_G1_TASKS_IMPORT_FILTER=cgc_topdown_curriculum_g1_29dof_dex3

RUN apt-get update && apt-get install -y --no-install-recommends \
    bash \
    ca-certificates \
    git \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libglu1-mesa-dev \
    libxt6 \
    libvulkan1 \
    vulkan-tools \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /opt/conda /opt/conda
COPY --from=builder /workspace/IsaacLab /workspace/IsaacLab
COPY --from=builder /workspace/project /workspace/project
COPY docker/entrypoint.sh /opt/enpm690/bin/entrypoint.sh

RUN chmod +x /opt/enpm690/bin/entrypoint.sh

WORKDIR /workspace/project

ENTRYPOINT ["/opt/enpm690/bin/entrypoint.sh"]
CMD ["bash"]
