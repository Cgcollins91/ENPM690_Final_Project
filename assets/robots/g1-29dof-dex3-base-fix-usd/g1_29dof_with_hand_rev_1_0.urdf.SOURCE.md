# Source: g1_29dof_with_hand_rev_1_0.urdf

Sourced from Unitree's official public repository:

- Repo: https://github.com/unitreerobotics/unitree_ros
- Path: `robots/g1_description/g1_29dof_with_hand_rev_1_0.urdf`
- Branch: master
- License: BSD-3-Clause (attribution preserved per LICENSE root of upstream repo)
- Date sourced: 2026-05-11

## Verified properties

- 7 right-arm movable joints: right_shoulder_pitch_joint, right_shoulder_roll_joint,
  right_shoulder_yaw_joint, right_elbow_joint, right_wrist_roll_joint,
  right_wrist_pitch_joint, right_wrist_yaw_joint.
- Right hand movable joints (7): right_hand_thumb_0_joint, right_hand_thumb_1_joint,
  right_hand_thumb_2_joint, right_hand_index_0_joint, right_hand_index_1_joint,
  right_hand_middle_0_joint, right_hand_middle_1_joint.
- right_hand_palm_joint is FIXED (palm is rigidly attached to right_wrist_yaw_link).
- right_hand_palm_link exists (EE frame for Lula IK).
- Root link: pelvis.
- floating_base_joint and "world" link are commented out in the upstream URDF.
