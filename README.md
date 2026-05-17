# ENPM690 Final Project

Chris Collins ENPM 690 Final Project Unitree G1/Dex3 top-down red-block
reach, grasp, and lift training pipeline in Isaac Sim / Isaac Lab.

The canonical launcher is:

```bash
python3 scripts/launch_pipeline.py
```

It resolves to `training/curriculum/pipeline.py`, which starts a fresh
adaptive BC/DAgger/FastTD3 run and launches the trainer through
`python -m training.native_entrypoint`.

## Contents

- `scripts/launch_pipeline.py`: canonical pipeline launcher.
- `scripts/eval_checkpoint.py`: headless checkpoint evaluation from a run
  manifest.
- `scripts/eval_visualization.py`: GUI checkpoint playback from a run manifest.
- `src/enpm690_final_project/`: typed run profiles, manifest generation, and
  subprocess planning.
- `training/`: modular training runtime.
- `tasks/`: Isaac Lab task registration and MDP code.
- `robots/` and `assets/`: Unitree G1/Dex3 robot config and required USD/URDF
  assets.
- `Dockerfile` and `docker/`: reproducible Isaac Sim / Isaac Lab container.
- `requirements.txt` and `pyproject.toml`: Python dependency metadata.

Generated outputs are not part of the package: `runs/`, `runs_training/`,
`checkpoints/`, TensorBoard events, JSONL logs, and `.pt` checkpoints.

## Submission Artifacts

- `Chris_Collins_ENPM_690_Final_Report.pdf`: final written report.
- Source code, robot assets, Dockerfile, and launch scripts are included in
  this repository.
- Trained checkpoints and full run logs are generated artifacts. If a grader
  needs to reproduce evaluation or visualization from an existing trained
  policy, provide the matching run directory separately with at least
  `manifest.json` and the referenced checkpoint, for example `latest.pt`.

## Git LFS Assets

If this repository was obtained through Git, fetch LFS-backed robot assets
before building or running. A source archive must contain real binary assets,
not Git LFS pointer files:

```bash
git lfs install
git lfs pull
test "$(wc -c < assets/robots/g1-29dof-dex3-base-fix-usd/g1_29dof_with_dex3_base_fix.usd)" -gt 1000000
```

## Docker

Docker is the recommended external-user path because it fixes the Isaac Sim,
Isaac Lab, PyTorch, and FastTD3 versions used by this project. The image build
still needs network access for apt, Miniconda, Python wheels, Isaac Lab, and
FastTD3.

Check GPU access:

```bash
nvidia-smi
docker run --rm --gpus all nvidia/cuda:12.2.0-runtime-ubuntu22.04 nvidia-smi
```

Build:

```bash
docker build -t enpm690-final-project:latest .
mkdir -p runs runs_training checkpoints
```

Rebuild this image after any source-tree change. The current image should show
`"/workspace/project"` as its working directory:

```bash
docker image inspect enpm690-final-project:latest \
  --format '{{.Created}} {{json .Config.WorkingDir}}'
```

Resolve the canonical command without launching Isaac:

```bash
docker run --gpus all --rm -it --network host \
  --user "$(id -u):$(id -g)" \
  -e ENPM690_PYTHON=/opt/conda/envs/unitree_sim_env/bin/python \
  -v "$PWD/runs:/workspace/project/runs" \
  -v "$PWD/runs_training:/workspace/project/runs_training" \
  -v "$PWD/checkpoints:/workspace/project/checkpoints" \
  enpm690-final-project:latest \
  python scripts/launch_pipeline.py \
    --run-dir runs_training/Final/pipeline_dry_run \
    --dry-run
```

Run a 100-step smoke test:

```bash
docker run --gpus all --rm -it --network host \
  --user "$(id -u):$(id -g)" \
  -e ENPM690_PYTHON=/opt/conda/envs/unitree_sim_env/bin/python \
  -v "$PWD/runs:/workspace/project/runs" \
  -v "$PWD/runs_training:/workspace/project/runs_training" \
  -v "$PWD/checkpoints:/workspace/project/checkpoints" \
  enpm690-final-project:latest \
  python scripts/launch_pipeline.py \
    --run-dir runs_training/Final/pipeline_smoke_100steps \
    --num-envs 1 \
    --total-steps 100 \
    --teacher-only-steps 10 \
    --dagger-steps 10 \
    --eval-every 0 \
    --checkpoint-every 100 \
    --log-every 10
```

Run the full pipeline:

```bash
docker run --gpus all --rm -it --network host \
  --user "$(id -u):$(id -g)" \
  -e ENPM690_PYTHON=/opt/conda/envs/unitree_sim_env/bin/python \
  -v "$PWD/runs:/workspace/project/runs" \
  -v "$PWD/runs_training:/workspace/project/runs_training" \
  -v "$PWD/checkpoints:/workspace/project/checkpoints" \
  enpm690-final-project:latest \
  python scripts/launch_pipeline.py \
    --run-dir runs_training/v35_canonical_fresh_fullreward_adaptive_bc_final_r5
```

The same workflows are available through Compose:

```bash
UID="$(id -u)" GID="$(id -g)" docker compose run --rm dry-run
UID="$(id -u)" GID="$(id -g)" docker compose run --rm smoke
UID="$(id -u)" GID="$(id -g)" docker compose run --rm train
```

## Local Install

Use local install only when Isaac Sim 5.1 and Isaac Lab are already installed in
the target Python environment. The Dockerfile is the reference package list.

```bash
python3 -m pip install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 \
  --index-url https://download.pytorch.org/whl/cu128
python3 -m pip install "isaacsim[all,extscache]==5.1.0" \
  --extra-index-url https://pypi.nvidia.com
python3 -m pip install -r requirements.txt
```

Install Isaac Lab in the same environment:

```bash
git clone https://github.com/isaac-sim/IsaacLab.git ../IsaacLab
cd ../IsaacLab
git checkout 80094be3245aa5c8376a7464d29cb4412ea518f5
./isaaclab.sh --install
cd -
```

Set runtime paths:

```bash
export PYTHONPATH="$PWD/src:$PWD:${PYTHONPATH:-}"
export UNITREE_TASKS_IMPORT_FILTER=tasks.g1_tasks.cgc_topdown_curriculum_g1_29dof_dex3
export UNITREE_G1_TASKS_IMPORT_FILTER=cgc_topdown_curriculum_g1_29dof_dex3
export ENPM690_PYTHON="$(command -v python3)"
```

If Isaac must be launched through a separate Python wrapper, point
`ENPM690_PYTHON` at that Isaac Sim Python executable:

```bash
export ENPM690_PYTHON="/path/to/isaac-sim/python.sh"
```

Create local output directories:

```bash
mkdir -p runs runs_training checkpoints
```

Resolve the command without launching Isaac:

```bash
python3 scripts/launch_pipeline.py \
  --run-dir runs_training/Final/pipeline_dry_run \
  --dry-run
```

Run local 100-step smoke test:

```bash
python3 scripts/launch_pipeline.py \
  --run-dir runs_training/Final/pipeline_smoke_100steps \
  --num-envs 1 \
  --total-steps 100 \
  --teacher-only-steps 10 \
  --dagger-steps 10 \
  --eval-every 0 \
  --checkpoint-every 100 \
  --log-every 10
```


Run the full pipeline locally:

```bash
python3 scripts/launch_pipeline.py \
  --run-dir runs_training/v35_canonical_fresh_fullreward_adaptive_bc_final_r5
```

## CLI

Common options:

- `--run-dir`: output directory.
- `--resume-from`: checkpoint to load.
- `--no-resume-replay`: resume model/optimizers without replay.
- `--no-resume-global-step`: restart step counters from zero.
- `--num-envs`, `--total-steps`, `--teacher-only-steps`, `--dagger-steps`.
- `--post-floor-rl-steps`, `--assist-floor`.
- `--eval-every`, `--checkpoint-every`, `--log-every`.
- `--tensorboard-dir`: TensorBoard event directory. Defaults to `<run-dir>/tb`.
- `--dry-run`: write `manifest.json` and print the resolved command only.

## Outputs

Each pipeline run writes:

- `manifest.json`: resolved command, environment, source hashes, git state, and
  run metadata.
- `stdout.log`: trainer/stdout trace.
- `log.jsonl`: structured metrics and diagnostics.
- `latest.pt`: latest replay-bearing checkpoint.
- `step_XXXXXX.pt`: rolling replay-bearing checkpoints.
- `final_replay.pt`: final handoff checkpoint.
- `tb/`: TensorBoard event files.

## TensorBoard

TensorBoard events are written to `<run-dir>/tb` unless `--tensorboard-dir`
overrides the location. Dry-runs write a manifest only and do not create useful
TensorBoard data.

Docker:

```bash
docker run --rm -it --network host \
  --user "$(id -u):$(id -g)" \
  -v "$PWD/runs_training:/workspace/project/runs_training" \
  enpm690-final-project:latest \
  python -m tensorboard.main \
    --logdir runs_training/v35_canonical_fresh_fullreward_adaptive_bc_final_r5/tb \
    --host 0.0.0.0 \
    --port 6006
```

Local:

```bash
python3 -m tensorboard.main \
  --logdir runs_training/v35_canonical_fresh_fullreward_adaptive_bc_final_r5/tb \
  --host localhost \
  --port 6006
```

Then open:

```text
http://localhost:6006
```

If port `6006` is busy, use another port such as `6007`.

## Evaluate a Checkpoint

`scripts/eval_checkpoint.py` evaluates a saved policy checkpoint using the
source training run's `manifest.json`. Relative checkpoint names resolve inside
`--source-run-dir`. These commands require a generated or separately provided
run directory; checkpoints are not tracked in the source package.

Docker:

```bash
docker run --gpus all --rm -it --network host \
  --user "$(id -u):$(id -g)" \
  -e ENPM690_PYTHON=/opt/conda/envs/unitree_sim_env/bin/python \
  -v "$PWD/runs:/workspace/project/runs" \
  -v "$PWD/runs_training:/workspace/project/runs_training" \
  -v "$PWD/checkpoints:/workspace/project/checkpoints" \
  enpm690-final-project:latest \
  python scripts/eval_checkpoint.py \
    --source-run-dir runs_training/v35_canonical_fresh_fullreward_adaptive_bc_final_r5 \
    --checkpoint latest.pt \
    --run-dir runs_training/v35_canonical_fresh_fullreward_adaptive_bc_final_r5/eval_latest \
    --episodes 5 \
    --steps 1000 \
    --num-envs 1 \
    --stream
```

Local:

```bash
python3 scripts/eval_checkpoint.py \
  --source-run-dir runs_training/v35_canonical_fresh_fullreward_adaptive_bc_final_r5 \
  --checkpoint latest.pt \
  --run-dir runs_training/v35_canonical_fresh_fullreward_adaptive_bc_final_r5/eval_latest \
  --episodes 5 \
  --steps 1000 \
  --num-envs 1 \
  --stream
```

Use `--dry-run` to print the resolved Isaac command without running evaluation.
Outputs are written under `--run-dir` and include `manifest.json`, `stdout.log`,
`log.jsonl`, `summary.json`, and `summary.jsonl`.

## Visualize a Checkpoint

`scripts/eval_visualization.py` replays a checkpoint in GUI mode using the
training run's `manifest.json`. The default mode is policy-only playback.

Docker with X11:

```bash
xhost +local:docker
docker run --gpus all --rm -it --network host \
  --user "$(id -u):$(id -g)" \
  -e DISPLAY="$DISPLAY" \
  -e ENPM690_PYTHON=/opt/conda/envs/unitree_sim_env/bin/python \
  -e NVIDIA_VISIBLE_DEVICES=all \
  -e NVIDIA_DRIVER_CAPABILITIES=compute,utility,video,graphics,display \
  -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
  -v /etc/vulkan/icd.d:/etc/vulkan/icd.d:ro \
  -v /usr/share/vulkan/icd.d:/usr/share/vulkan/icd.d:ro \
  -v "$PWD/runs:/workspace/project/runs" \
  -v "$PWD/runs_training:/workspace/project/runs_training" \
  -v "$PWD/checkpoints:/workspace/project/checkpoints" \
  enpm690-final-project:latest \
  python scripts/eval_visualization.py \
    --run-dir runs_training/v35_canonical_fresh_fullreward_adaptive_bc_final_r5 \
    --checkpoint latest.pt \
    --mode policy \
    --episodes 3 \
    --steps 1000 \
    --num-envs 1
```

Local:

```bash
python3 scripts/eval_visualization.py \
  --run-dir runs_training/v35_canonical_fresh_fullreward_adaptive_bc_final_r5 \
  --checkpoint latest.pt \
  --mode policy \
  --episodes 3 \
  --steps 1000 \
  --num-envs 1
```

The visualization output defaults to
`<run-dir>/eval_visualization_<checkpoint-stem>/`. Use `--mode teacher` for
teacher-only playback, or `--mode mixed --teacher-assist-mix 0.5` for mixed
teacher/policy playback. Use `--dry-run` to write the visualization manifest
and print the resolved Isaac command without launching the GUI.

## License

Project-specific coursework material is covered by `LICENSE`. Third-party
notices for Unitree, Isaac Lab / Isaac Sim, FastTD3, PyTorch, and Python
dependencies are in `NOTICE`.
