from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
STATE_MACHINE_PATH = (
    REPO_ROOT
    / "tasks"
    / "g1_tasks"
    / "cgc_topdown_curriculum_g1_29dof_dex3"
    / "mdp"
    / "state_machine.py"
)


def _load_state_machine():
    spec = importlib.util.spec_from_file_location("topdown_state_machine_for_test", STATE_MACHINE_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _Scene(dict):
    pass


def _rigid(pos: torch.Tensor) -> SimpleNamespace:
    quat = torch.zeros((pos.shape[0], 4), dtype=pos.dtype, device=pos.device)
    quat[:, 0] = 1.0
    return SimpleNamespace(data=SimpleNamespace(root_pos_w=pos, root_quat_w=quat))


def _env(num_envs: int, *, visible_sources: bool = True) -> SimpleNamespace:
    device = torch.device("cpu")
    env = SimpleNamespace(num_envs=num_envs, device=device, scene=_Scene())
    env._topdown_use_visible_source_objects = visible_sources
    env._topdown_source_pose_idx = torch.arange(num_envs, device=device) % 3
    return env


def test_visible_source_block_pose_gathers_per_env_active_object() -> None:
    state_machine = _load_state_machine()
    env = _env(3)
    env.scene["object"] = _rigid(torch.tensor([[10.0, 0.0, 0.0], [11.0, 0.0, 0.0], [12.0, 0.0, 0.0]]))
    env.scene["object_yellow"] = _rigid(torch.tensor([[20.0, 0.0, 0.0], [21.0, 0.0, 0.0], [22.0, 0.0, 0.0]]))
    env.scene["object_blue"] = _rigid(torch.tensor([[30.0, 0.0, 0.0], [31.0, 0.0, 0.0], [32.0, 0.0, 0.0]]))

    pos, _ = state_machine._block_pose(env)

    assert torch.equal(pos[:, 0], torch.tensor([10.0, 21.0, 32.0]))


def test_visible_source_contact_forces_gather_per_env_active_filter() -> None:
    state_machine = _load_state_machine()
    env = _env(3)
    force_matrix = torch.zeros((3, 1, 3, 3), dtype=torch.float32)
    force_matrix[:, 0, :, 0] = torch.tensor(
        [
            [1.0, 100.0, 100.0],
            [100.0, 2.0, 100.0],
            [100.0, 100.0, 3.0],
        ]
    )

    forces = state_machine._active_filter_forces(env, force_matrix)

    assert torch.equal(forces[:, 0], torch.tensor([1.0, 2.0, 3.0]))


def test_single_object_contact_forces_keep_existing_sum_behavior() -> None:
    state_machine = _load_state_machine()
    env = _env(2, visible_sources=False)
    force_matrix = torch.zeros((2, 1, 3, 3), dtype=torch.float32)
    force_matrix[:, 0, :, 0] = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

    forces = state_machine._active_filter_forces(env, force_matrix)

    assert torch.equal(forces[:, 0], torch.tensor([6.0, 15.0]))
