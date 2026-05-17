# Training Package

`training/` is the authoritative native Isaac Lab / FastTD3 training
implementation. Training, checkpointing, evaluation, and logging work should use
this package.

## Current Final Pipeline

The final-project training path is:

```text
scripts/launch_pipeline.py
  -> training/curriculum/pipeline.py
  -> src/enpm690_final_project/training_engine.py
  -> python -m training.native_entrypoint
  -> training/native/
```

The pipeline launcher applies the full-reward adaptive BC/DAgger contract:

1. 150k teacher-only replay rows.
2. 500k noisy assisted DAgger/BC rows with clean teacher labels.
3. TD/FastTD3 switch at 650k with replay preserved.
4. Strict-contact-gated adaptive assist decay.
5. Teacher BC/relabeling disabled after assist reaches the floor.
6. 1M policy-only RL rows after the floor before stopping.

See the root `README.md` for the user-facing Docker, smoke-test, resume, and
TensorBoard commands.

## Package Layout

- `actions/`: action layout, action mixing, action gates, action buffers,
  policy-arm helpers, and action-level losses.
- `core/`: CLI parsing, runtime configs, task routing, context objects, and
  backend dispatch.
- `curriculum/`: final curriculum launch contracts, including `pipeline.py`.
- `env/`: observation flattening, startup contracts, Isaac startup boundaries,
  terminal-observation handling, and reset cleanup.
- `eval/`: eval action selection, rollout state, success overrides, summary
  rows, eval logging, and post-eval reset helpers.
- `geometry/`: geometric math, block/tip diagnostics, top-down metrics, IK
  masks, IK solvers, lift latch logic, in-pocket helpers, and source-conditioned
  rows.
- `io/`: checkpoint IO, checkpoint scheduling, compatibility checks, replay
  handoff handling, warm starts, and finalization.
- `logging/`: JSONL rows, progress lines, diagnostics, reward/termination
  manager readers, TensorBoard helpers, and episode-end formatting.
- `model/`: TD3 agents, networks, normalization, backend selection, and optional
  upstream FastTD3 loading.
- `native/`: callback-driven Isaac backend, live loop, eval, events, native
  teacher providers, startup, and bounded Isaac smoke entrypoints.
- `state/`: episode state, loop planning, cadence, replay buffers, n-step
  transition collection, seeding, phase overrides, pre-roll state, and update
  scheduling.
- `teacher/`: scripted teacher actions, closure schedules, teacher arm control,
  teacher IK state, contact teacher logic, contact pre-roll actions, reach
  signals, and curriculum gates.



## Direct Trainer Usage

The pipeline launcher should be used for final runs because it pins reward
weights, support bounds, BC/relabel settings, assist scheduling, checkpointing,
and success gates. Direct `python -m training` launches remain useful for
low-level smoke tests and focused debugging. Run direct trainer commands only
after completing the root Docker or local setup and using the same Python
environment that can import Isaac Sim, Isaac Lab, and this project's task
modules:

```bash
python -m training --task Isaac-Topdown-Curriculum-G129-Dex3-Joint \
  --num-envs 1 \
  --total-steps 1 \
  --start-steps 0 \
  --headless \
  --eval-steps 0 \
  --eval-episodes 0 \
  --checkpoint-every 0 \
  --rolling-checkpoint-every 0 \
  --tensorboard-dir off
```
