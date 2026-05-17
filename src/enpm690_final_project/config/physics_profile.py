"""Pure-Python physics profile resolver for the G1+Dex3 topdown env cfg.

Reads the ``TOPDOWN_PHYSICS_PROFILE`` env var plus the ten per-knob
override env vars (two of which have identical values in both profiles but
retain existing env-var pathways); returns a fully resolved
``ResolvedPhysicsProfile`` dataclass that the env cfg can apply directly
without further branching.

Profiles:
  * ``default`` (or absent / unknown) — project task values.
  * ``nvidia_mirror`` — a lower-substep, lighter-block, larger-contact-offset
    profile aligned with common Isaac Sim pick-place defaults.

User env vars always win over profile defaults (precedence preserves
existing override pathways). Pure Python — no Isaac Lab imports, so this
module is testable from a plain Python invocation.

The physics profile layer exists because material/solver defaults have a large
effect on whether "contact" means a real cage, a bounce, or a sliding nudge.
Profiles capture coherent sets of physics assumptions, while individual
``TOPDOWN_*`` env vars remain available for one-off ablations and smoke tests.
"""

from __future__ import annotations  # keeps annotations lazy for forward references

from dataclasses import dataclass  # imports dataclass helpers used by config groups
from typing import Mapping  # imports typing helpers used by config annotations

VALID_PROFILES = ("default", "nvidia_mirror")  # lists physics profile names accepted by the launcher


@dataclass(frozen=True)  # makes the following config group immutable
class ResolvedPhysicsProfile:  # defines the resolved physics profile config group
    """All 18 physics knobs after profile + env-var resolution."""

    profile_name: str  # Records which physics profile supplied the resolved values
    # PhysX scene
    num_substeps                 : int  # Sets PhysX substeps per simulation step
    num_position_iterations      : int  # Sets PhysX position solver iterations
    num_velocity_iterations      : int  # Sets PhysX velocity solver iterations
    contact_offset               : float  # Sets the collision contact generation distance
    rest_offset                  : float  # Sets the collision rest separation distance
    friction_correlation_distance: float  # Sets the contact patch friction correlation distance
    physics_dt                   : float  # Sets the simulation timestep used by the physics scene
    # Block rigid body / geometry
    block_dynamic: bool  # Controls whether the target block is simulated as a dynamic rigid body
    block_mass   : float  # Sets the target block mass in kilograms
    block_size   : float  # Sets the target block edge length in meters
    # Block material
    block_static_friction         : float  # Sets static friction for the target block material
    block_dynamic_friction        : float  # Sets dynamic friction for the target block material
    block_friction_combine_mode   : str  # Selects how block friction combines with contacting materials
    block_restitution_combine_mode: str  # Selects how block restitution combines with contacting materials
    # Fingertip material
    fingertip_static_friction         : float  # Sets static friction for the fingertip material
    fingertip_dynamic_friction        : float  # Sets dynamic friction for the fingertip material
    fingertip_friction_combine_mode   : str  # Selects how fingertip friction combines with the block material
    fingertip_restitution_combine_mode: str  # Selects how fingertip restitution combines with the block material

    def as_log_line(self) -> str:  # formats resolved physics knobs for one-line stdout logging
        """Single-line ``key=value`` summary for stdout logging."""
        return (  # returns the multi-line expression below
            "[physics-profile] "  # adds literal text to the surrounding expression
            f"profile={self.profile_name} "  # adds formatted data to the returned text
            f"substeps={self.num_substeps} "  # adds formatted data to the returned text
            f"pos_iters={self.num_position_iterations} "  # adds formatted data to the returned text
            f"vel_iters={self.num_velocity_iterations} "  # adds formatted data to the returned text
            f"contact_offset={self.contact_offset} "  # adds formatted data to the returned text
            f"rest_offset={self.rest_offset} "  # adds formatted data to the returned text
            f"friction_corr_dist={self.friction_correlation_distance} "  # adds formatted data to the returned text
            f"physics_dt={self.physics_dt} "  # adds formatted data to the returned text
            f"block_dynamic={self.block_dynamic} "  # adds formatted data to the returned text
            f"block_mass={self.block_mass} "  # adds formatted data to the returned text
            f"block_size={self.block_size} "  # adds formatted data to the returned text
            f"block_fric={self.block_static_friction}/{self.block_dynamic_friction} "  # adds formatted data to the returned text
            f"block_fric_combine={self.block_friction_combine_mode} "  # adds formatted data to the returned text
            f"block_rest_combine={self.block_restitution_combine_mode} "  # adds formatted data to the returned text
            f"finger_fric={self.fingertip_static_friction}/{self.fingertip_dynamic_friction} "  # adds formatted data to the returned text
            f"finger_fric_combine={self.fingertip_friction_combine_mode} "  # adds formatted data to the returned text
            f"finger_rest_combine={self.fingertip_restitution_combine_mode}"  # adds formatted data to the returned text
        )  # closes the current expression


# Default-profile values mirror the literals currently in cgc_topdown_curriculum_env_cfg dot py
_DEFAULT_VALUES = {  # defines baseline physics values for the default profile
    "num_substeps": 4,  # Sets the default num substeps physics value
    "num_position_iterations": 16,  # Sets the default num position iterations physics value
    "num_velocity_iterations": 4,  # Sets the default num velocity iterations physics value
    "contact_offset": 0.002,  # Sets the default contact offset physics value
    "rest_offset": 0.0,  # Sets the default rest offset physics value
    "friction_correlation_distance": 0.002,  # Sets the default friction correlation distance physics value
    "physics_dt": 0.005,  # Sets the default physics dt physics value
    "block_dynamic": False,  # Sets the default block dynamic physics value
    "block_mass": 0.25,  # Sets the default block mass physics value
    "block_size": 0.08,  # Sets the default block size physics value
    "block_static_friction": 10.0,  # Sets the default block static friction physics value
    "block_dynamic_friction": 1.5,  # Sets the default block dynamic friction physics value
    "block_friction_combine_mode": "max",  # Sets the default block friction combine mode physics value
    "block_restitution_combine_mode": "min",  # Sets the default block restitution combine mode physics value
    "fingertip_static_friction": 3.0,  # Sets the default fingertip static friction physics value
    "fingertip_dynamic_friction": 2.5,  # Sets the default fingertip dynamic friction physics value
    "fingertip_friction_combine_mode": "max",  # Sets the default fingertip friction combine mode physics value
    "fingertip_restitution_combine_mode": "min",  # Sets the default fingertip restitution combine mode physics value
}  # closes the current expression

# nvidia_mirror overlays: only the fields that differ from defaults
_NVIDIA_MIRROR_OVERLAY = {  # defines physics values changed by the NVIDIA mirror profile
    "num_substeps": 1,  # Overrides num substeps for the NVIDIA mirror physics profile
    "num_position_iterations": 8,  # Overrides num position iterations for the NVIDIA mirror physics profile
    "num_velocity_iterations": 1,  # Overrides num velocity iterations for the NVIDIA mirror physics profile
    "contact_offset": 0.02,  # Overrides contact offset for the NVIDIA mirror physics profile
    "friction_correlation_distance": 0.025,  # Overrides friction correlation distance for the NVIDIA mirror physics profile
    "block_dynamic": True,  # Overrides block dynamic for the NVIDIA mirror physics profile
    "block_mass": 0.05,  # Overrides block mass for the NVIDIA mirror physics profile
    "block_static_friction": 0.5,  # Overrides block static friction for the NVIDIA mirror physics profile
    "block_dynamic_friction": 0.5,  # Overrides block dynamic friction for the NVIDIA mirror physics profile
    "block_friction_combine_mode": "average",  # Overrides block friction combine mode for the NVIDIA mirror physics profile
    "block_restitution_combine_mode": "average",  # Overrides block restitution combine mode for the NVIDIA mirror physics profile
    "fingertip_static_friction": 0.5,  # Overrides fingertip static friction for the NVIDIA mirror physics profile
    "fingertip_dynamic_friction": 0.5,  # Overrides fingertip dynamic friction for the NVIDIA mirror physics profile
    "fingertip_friction_combine_mode": "average",  # Overrides fingertip friction combine mode for the NVIDIA mirror physics profile
    "fingertip_restitution_combine_mode": "average",  # Overrides fingertip restitution combine mode for the NVIDIA mirror physics profile
}  # closes the current expression

# Mapping of dataclass field name -> env var name + parser Includes every
# field whose value is env-var-controllable in env_cfg (block_size and
# rest_offset have identical values in both profiles but retain their existing
# env-var pathways) Combine modes and PhysX iters are profile-direct
# (not env-overridable) by design
_ENV_OVERRIDABLE: dict[str, tuple[str, str]] = {  # records env vars that may override resolved physics values
    "contact_offset": ("TOPDOWN_CONTACT_OFFSET", "float"),  # Lets TOPDOWN_CONTACT_OFFSET override the resolved contact offset value
    "rest_offset": ("TOPDOWN_REST_OFFSET", "float"),  # Lets TOPDOWN_REST_OFFSET override the resolved rest offset value
    "friction_correlation_distance": ("TOPDOWN_FRICTION_CORRELATION_DISTANCE", "float"),  # Lets TOPDOWN_FRICTION_CORRELATION_DISTANCE override the resolved friction correlation distance value
    "block_dynamic": ("TOPDOWN_DYNAMIC_BLOCK", "bool"),  # Lets TOPDOWN_DYNAMIC_BLOCK override the resolved block dynamic value
    "block_mass": ("TOPDOWN_BLOCK_MASS", "float"),  # Lets TOPDOWN_BLOCK_MASS override the resolved block mass value
    "block_size": ("TOPDOWN_BLOCK_SIZE", "float"),  # Lets TOPDOWN_BLOCK_SIZE override the resolved block size value
    "block_static_friction": ("TOPDOWN_BLOCK_STATIC_FRICTION", "float"),  # Lets TOPDOWN_BLOCK_STATIC_FRICTION override the resolved block static friction value
    "block_dynamic_friction": ("TOPDOWN_BLOCK_DYNAMIC_FRICTION", "float"),  # Lets TOPDOWN_BLOCK_DYNAMIC_FRICTION override the resolved block dynamic friction value
    "fingertip_static_friction": ("TOPDOWN_FINGERTIP_STATIC_FRICTION", "float"),  # Lets TOPDOWN_FINGERTIP_STATIC_FRICTION override the resolved fingertip static friction value
    "fingertip_dynamic_friction": ("TOPDOWN_FINGERTIP_DYNAMIC_FRICTION", "float"),  # Lets TOPDOWN_FINGERTIP_DYNAMIC_FRICTION override the resolved fingertip dynamic friction value
}  # closes the current expression


def _parse_bool(raw: str) -> bool:  # parses legacy env-var boolean text
    return raw.strip().lower() not in {"0", "false", "no", "off", ""}  # returns the computed value


def resolve_physics_profile(env: Mapping[str, str]) -> ResolvedPhysicsProfile:  # resolves profile defaults and env overrides into physics knobs
    """Resolve all 18 physics knobs from a profile + env override mapping.

    Args:
        env: Mapping (e.g. ``os.environ``) of env-var name to string value.
            Pass ``{}`` to get pure default-profile values.

    Returns:
        ``ResolvedPhysicsProfile`` with every field populated. Unknown
        profile names fall back to the default branch (no exception).
    """
    requested = env.get("TOPDOWN_PHYSICS_PROFILE", "default").strip().lower()  # Reads the requested physics profile name from env
    profile_name = requested if requested in VALID_PROFILES else "default"  # Falls back to default when the requested physics profile is unknown

    values = dict(_DEFAULT_VALUES)  # Starts resolved physics values from the default profile
    if profile_name == "nvidia_mirror":  # Checks whether profile name == "nvidia mirror"
        values.update(_NVIDIA_MIRROR_OVERLAY)  # merges override values into the current mapping

    for field_name, (env_var, kind) in _ENV_OVERRIDABLE.items():  # iterates over configured values
        raw = env.get(env_var, "")  # Reads the raw env-var override for this physics field
        if raw == "":  # Checks whether raw == ""
            continue  # skips this item and continues validation
        if kind == "float":  # Checks whether kind == "float"
            values[field_name] = float(raw)  # stores the resolved value in the mapping
        elif kind == "bool":  # Checks alternate branch for kind == "bool"
            values[field_name] = _parse_bool(raw)  # stores the resolved value in the mapping
        else:  # pragma: no cover - defensive branch outside normal profile parser inputs
            raise ValueError(f"unknown kind {kind!r} for {field_name}")  # raises an error for invalid config state

    return ResolvedPhysicsProfile(profile_name=profile_name, **values)  # returns the fully resolved physics profile dataclass
