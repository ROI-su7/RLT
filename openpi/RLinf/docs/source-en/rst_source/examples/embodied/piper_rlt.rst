Piper Dual-Arm Real-World RLT Adaptation
========================================

This adaptation uses the RLinf RLT implementation as its algorithmic baseline
and adds the ROS 1, three-camera, 14-dimensional absolute-joint action, safety
limit, evdev keyboard, and two-host Ray support required by Cobot Magic.

Key conventions
---------------

- Stage 1 freezes the native-JAX ``pi05_connect_strings`` checkpoint and uses
  eight-GPU FSDP to optimize only the RL-token encoder/decoder reconstruction
  loss. Its checkpoint records the exact JAX config, image-only prefix contract,
  fp32 master weights and encoder fingerprint.
- For a clean Stage 2 run, collect 1,000 transitions before any learner
  update. Keep ``b`` on the JAX reference until 3,000 learner updates have
  completed.
- After warmup, each new transition creates only five learner-update credits.
  With no new data, learning waits. The critic-to-actor update ratio is 2:1.
- Stage 2 uses sparse binary terminal rewards: ``c=+1`` and ``x/q=0``.
- ``b`` marks the next complete 10-action critical chunk. During warmup, the
  JAX reference still controls the robot and the transition is recorded; the
  actor takes over only after learner readiness.
- The online actor matches RL-Token-Pi05-open's full-output Gaussian contract:
  ``mu=MLP(z,p,a_ref)`` with random initialization and fixed normalized
  ``sigma=0.002``. The existing two-layer ``Linear-ReLU-Linear`` network
  (hidden width 256) remains checkpoint-compatible. Smoothing follows
  Yyshadow/openpi-RLT: ``bc_weight * BC - 0.1 * Q1 + 10 * delta_penalty``.
  BC weight is 10 before update 3000 and 5 thereafter; Actor LR is 1e-4,
  including checkpoint resume. Training keeps 50%
  reference dropout. SAC training still samples from this distribution; robot
  execution uses its mean. The actor retains a full-output head.
  ``algorithm.rlt_schedule.actor_direct_execution: true`` disables Actor
  residual EMA/taper, executor EMA, chunk-boundary blending and RTC.
  Reference-relative bounds remain active: 0.01 rad per arm joint and 0.0015 m
  per gripper. Actor windows start at reference step 0. BC constrains raw
  samples; actor Q and target-actor bootstrap query the bounded
  actions. ``delta_penalty`` compares physical adjacent arm increments to
  reference/human increments on both six-joint arms, excluding grippers,
  padding and source-switch pairs. It supervises raw outputs beyond bounds.
  Cross-chunk correction EMA has been removed. The restored command profile
  bounds accepted goals by 0.004 rad / 0.0002 m per policy tick, relative to
  the previous accepted goal. The 200-Hz tracker limits velocity/acceleration
  to 0.10 rad/s / 0.6 rad/s2 for arms and 0.004 m/s / 0.02 m/s2 for grippers;
  response time is 0.12 s. It carries command state across chunks and resets
  on HOLD. It limits complete Actor targets, so it can slow the reference
  component as well. Q's shared transform models correction bounds, not the
  stateful command profile. JAX keeps its configured EMA, RTC bridge and alignment, including
  during warmup. Without Actor RTC, inference latency can create a pause at
  chunk boundaries; direct execution alone does not guarantee smoothness.
  There is no whole-chunk deviation fallback. The critic
  bootstrap uses the existing EMA target actor and the minimum of the target
  twin-Q values. The target model keeps C's existing
  every-update Polyak cadence.
- This adopts openpi-RLT's physical-unit delta loss,
  not its projected LayerNorm/GELU network. Deployment remains deterministic;
  acceleration tracking is a local extension, not upstream B's default.
  Command smoothing does not reduce the 15-Hz correction update frequency;
  slower correction scheduling would be a separate implementation.
- Replay samples uniformly, without source quotas. ``[RLT correction]`` logs
  distinguish ACTOR, JAX_REFERENCE, HUMAN and mixed chunks. They compare
  requested actions against the same-time JAX window after RTC alignment,
  excluding invalid and human steps. Left/right arm statistics use radians;
  grippers use meters. Values include downstream blending/target limits and
  are not measured motor positions or proof of improved task success.
- Every two critic updates produce exactly one online-actor update. Partial
  transitions mask padded rewards and use ``gamma ** executed_steps``; full
  chunks retain ``gamma ** 10``.
- Policy targets are generated at 15 Hz. An isolated control process publishes
  interpolated commands at 200 Hz. The four-CAN configuration uses a bounded
  RTC suffix during JAX inference; HOLD/T invalidates the suffix and I replans.
- Camera frames use the same OpenPI PIL ``resize_with_pad`` path as
  ``examples/piper_real``. Do not replace it with OpenCV interpolation: the
  resulting pixel drift caused a repeatable first-slot reference offset on the
  real robot.
- Joint boundaries and per-step joint/gripper limits remain active. ``q`` is
  not a physical emergency stop.

See ``COBOT_MAGIC_RLT_DEPLOYMENT.md`` at the repository root for checkpoint
conversion, Stage 1 and Stage 2 configuration, two-host launch commands, and
the hardware-safety workflow.

Four-CAN takeover for Stage 2
----------------------------

The default Piper Stage-2 configs reuse the existing OpenPI supervised launch
without modifying its driver, role switching, two-second takeover delay or HOME:

.. code-block:: bash

   roslaunch piper start_inference_takeover.launch \
     master_left_can:=can_l_leader master_right_can:=can_r_leader \
     puppet_left_can:=can_l_follower puppet_right_can:=can_r_follower \
     publish_hz:=200

RLinf is the keyboard owner; do not run the standalone OpenPI client or
``piper_mode_keyboard.py`` alongside it. Policy commands go only to
``/teleop/policy/joint_left/right``. The original mux alone commands followers
via ``/teleop/command/joint_left/right``. The isolated executor clears pending
commands before publishing ``/teleop/policy_hold_ack``.

The Stage 2 launch scripts automatically select the only evdev keyboard that
provides all control keys. If several candidates are present, set
``RLINF_KEYBOARD_DEVICE=/dev/input/by-id/...`` to the intended device. Do not
depend on a ``/dev/input/eventN`` number, which can change after a reboot or USB
reconnection.

Start the robot-local action service with:

.. code-block:: bash

   cd /home/agilex/FluxVla/openpi/RLinf
   bash examples/embodiment/serve_piper_rlt_compact.sh

The launcher uses ``pi05_connect_strings`` and its config-defined prompt,
loads the same native checkpoint and frozen RL token, then warms both model
paths before listening on port 6006. It opens no ROS/CAN interface. The exact
action prefix is encoded locally: an 8192-byte ``rlt_z_rl`` replaces the
3145728-byte raw prefix over the cross-host link. Stride observations run as
single-item low-priority jobs so queued action requests can go first. The
preflight requires ``rlt_action_z_rl`` metadata in the four-CAN backend.

RTC remains gated by confirmed policy ownership and observation generation;
the executor drains its suffix before aligning the next actor window. Space,
T and terminal HOLD clear pending RTC work, including while inference is busy.

- ``I`` starts an idle episode or resumes inference after HOLD/human control.
- ``B`` opens collection from the next complete chunk; learner readiness
  decides whether JAX or the actor controls the robot. Actor corrections are
  bounded per dimension; there is no whole-chunk deviation fallback.
- ``T`` after B requests human control. Only confirmed ``TAKEOVER/ACTIVE`` steps
  carry intervention flags; mux-selected ``/master/joint_left/right`` targets
  are recorded as human actions, in radians/metres.
- ``C`` succeeds with reward +1; ``X`` fails with reward 0. Both finish and HOLD.
- Space requests the original software HOLD, retaining the episode and B gate.
  HOLD frames and obsolete chunk suffixes are not replayed; I replans.
- ``H`` is accepted in idle/HOLD and calls mux HOME, returning all four arms to this ROS
  launch's captured poses, then holding. HOME is not recorded. ``Q`` remains a
  zero-reward terminal HOLD alias. ``A/R/G/F`` have no function in this backend.

Validate ``I-B-C``, ``I-B-X``, ``I-B-T-C`` and ``I-B-T-X`` independently.
Human and autonomous data share ``logs/<run>/replay_buffer/rank_0``. Human
steps set ``intervene_flags``; warmup steps retain ``reference_warmup``. These
sampling tags can overlap. Terminal padding has an ``action_valid_mask`` and
does not contribute to BC; windows crossing discarded HOLD actions are dropped.
Both four-arm and follower-only modes emit ``rlt_keyboard_terminal`` on the
first C/X after B, allowing a late chunk-boundary event to label the last real
transition. Repeated terminal frames and episodes without B emit no such label.

Resuming a single-host Stage 2 run
----------------------------------

By default, the single-host launcher starts with fresh Stage 2 weights and
replay. Set both ``RLINF_STAGE2_RESUME_DIR`` and ``RLINF_REPLAY_LOAD_PATH`` to
resume weights while retaining all collected transitions. After the
old run has stopped, merge its complete replay with later collection runs;
use a new destination so source buffers remain intact:

.. code-block:: bash

   cd /home/agilex/FluxVla/openpi/RLinf
   source .venv-rlt/bin/activate
   python toolkits/merge_rlt_replay_buffers.py \
     --destination /absolute/path/to/merged/rank_0 \
     /absolute/path/to/first/replay_buffer/rank_0 \
     /absolute/path/to/second/replay_buffer/rank_0
   export RLINF_REPLAY_LOAD_PATH=/absolute/path/to/merged/rank_0
   export RLINF_STAGE2_RESUME_DIR=/absolute/path/to/checkpoints/global_step_50
   bash examples/embodiment/run_piper_rlt_stage2_single.sh

Verify ``skipped_missing_index_entries=0`` and the expected transition total
before restarting. A checkpoint by itself restores only its older replay
snapshot. During the first resumed B windows, ``replay/actor_switch_rate=0``
can mean the actor is ready but the action-deviation gate selected JAX; keep
collecting reference or human transitions and review the actor-reference
error before relaxing the gate.

Replay-only Stage 2 retraining
------------------------------

Use the replay-only entry point to retrain a fresh Actor/Critic from a persisted
``rank_0`` buffer. It launches only ``ActorGroup``: no ``EnvGroup``,
``RolloutGroup``, JAX client, ROS node, camera or CAN interface is constructed.
The source replay is read-only because auto-save is disabled; checkpoints still
contain replay snapshots for complete recovery.

.. code-block:: bash

   cd /home/agilex/FluxVla/openpi/RLinf
   source .venv-rlt/bin/activate

   export RLINF_REPLAY_LOAD_PATH=/absolute/path/to/replay_buffer/rank_0
   export RLINF_STAGE1_RLT_TOKEN=/absolute/path/to/rlt_token.pt

   bash examples/embodiment/run_piper_rlt_stage2_replay_only.sh

The launcher starts from random Stage 2 Actor/Critic weights, runs 2,000 global
learner steps by default, and saves every 50 steps under
``logs/<timestamp>-realworld_piper_rlt_stage2_replay_only/``. Do not set
``runner.resume_dir`` for this scratch-retraining workflow. Ray must already be
running on the local GPU node. Set ``RLINF_REPLAY_ONLY_MAX_STEPS`` to choose a
different target, for example ``10000``.

Legacy opt-in mode-2 leader takeover
-----------------------------------

The existing mode-0 and mode-1 ROS paths remain unchanged. Leader-follower
takeover is selected only by
``realworld_piper_rlt_stage2_ac_mlp_mode2.yaml`` (single host) or
``realworld_piper_rlt_stage2_ac_mlp_dual_mode2.yaml`` (two hosts). Start the
composite ROS driver with:

.. code-block:: bash

   roslaunch piper start_ms_piper.launch mode:=2 auto_enable:=true \
     mode2_leaders_preisolated:=true

The launch internally reuses the proven mode-1 follower position driver, but
the explicit mode-2 branch adds the isolated leader CAN bridge. Existing
mode-0 and mode-1 conditions and behavior are unchanged. Command ownership is:

``policy -> H/follower hold -> M/slow leader approach -> manual fine alignment
-> READY -> T/human teleop``.

The installed Piper API does not expose arbitrary position commands for a
teaching-input leader. Therefore ``M`` starts a monitored **manual** approach:
the operator moves the powered leaders slowly while the followers hold. The
state machine first requires ``approach_tolerance_rad``, then the tighter
``position_tolerance_rad`` for consecutive samples before it announces
``READY``. Missing/stale feedback, follower drift, or timeout fails closed.

At ``T``, the policy/RTC publisher is suspended before the independent bridge
is enabled. The bridge is then the only publisher to ``/master/joint_left`` and
``/master/joint_right`` and forwards shifted leader frames at 200 Hz. The
environment gathers observations without publishing a second command; measured
leader actions are stored as ``intervene_action``. ``C`` records success and
``X`` records failure, but both enter an unrecorded teleoperation return. Move
the leader back to the captured start (the follower follows), then press ``R``.
``R`` validates both leader and follower poses, disables the bridge, holds the
followers, and only then permits the next ``A``. It does not automatically move
the leader.

Outside mode 2, native ``0x15x`` linkage lets leaders control followers without
ROS. Piper S-V1.8-8 acknowledges role-neutral 0x470 offset commands but does
not apply them, and role-setting 0x470 commands are broadcast to both arms on
the shared CAN bus. Therefore software-only online ``0x15x``/``0x16x``
switching is unsupported on this wiring. Mode 2 fails closed unless both
leaders were physically isolated, configured, and verified at ``0x16x`` first.
Restoring ``0x15x`` likewise requires followers to be disconnected. See
``COBOT_MAGIC_RLT_DEPLOYMENT.md`` for the verified offline procedure.

Shared-CAN offline transition
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To enter mode 2 from native ``0x15x``:

1. Use native linkage to place both leader-follower pairs in a nearby safe
   pose. Stop RLT, RTC, policy, and all Piper ROS nodes, then power off all four
   arms.
2. With all arms powered off, physically disconnect both follower aviation/CAN
   plugs. Keep both followers disconnected and unpowered.
3. Power only the leaders, configure the CAN interfaces, and then configure
   each leader:

   .. code-block:: bash

      cd /home/agilex/cobot_magic/Piper_ros_private-ros-noetic
      source /opt/ros/noetic/setup.bash
      bash can_config.sh

      cd /home/agilex/FluxVla/openpi/RLinf

      /usr/bin/python3 \
        toolkits/realworld_check/configure_piper_mode2_master_offset.py \
        --can can_left --apply --i-confirm-follower-disconnected

      /usr/bin/python3 \
        toolkits/realworld_check/configure_piper_mode2_master_offset.py \
        --can can_right --apply --i-confirm-follower-disconnected

4. Do not power-cycle before checking the real emitted IDs:

   .. code-block:: bash

      timeout 3 candump can_left,160:7F0
      timeout 3 candump can_left,150:7F0
      timeout 3 candump can_right,160:7F0
      timeout 3 candump can_right,150:7F0

   Both ``0x16x`` filters must show leader frames, while both ``0x15x`` filters
   must remain empty. A ``timeout`` exit status of 124 is expected; judge the
   result by the emitted IDs. Stop if either side fails.
5. Power off the leaders. With all four arms powered off, reconnect the
   followers without moving any arm, then power all four arms.
6. Start ROS only after the preceding checks:

   .. code-block:: bash

      cd /home/agilex/cobot_magic/Piper_ros_private-ros-noetic
      source /opt/ros/noetic/setup.bash
      bash can_config.sh
      source devel/setup.bash
      export ROS_MASTER_URI=http://192.168.209.107:11311
      export ROS_IP=192.168.209.107

      roslaunch piper start_ms_piper.launch \
        mode:=2 \
        auto_enable:=true \
        mode2_leaders_preisolated:=true

7. Before starting the policy, verify both leader topics and the guarded mode-2
   services:

   .. code-block:: bash

      rostopic hz /rlinf/mode2/leader_left
      rostopic hz /rlinf/mode2/leader_right
      rosservice list | grep /rlinf/mode2

The ``mode2_leaders_preisolated`` flag is an operator assertion, not an address
switch. Never set it while the leaders still emit native ``0x15x`` frames.

To restore native linkage, stop all control processes, power off all arms, and
disconnect both followers again. Power only the leaders and run:

.. code-block:: bash

   cd /home/agilex/cobot_magic/Piper_ros_private-ros-noetic
   source /opt/ros/noetic/setup.bash
   bash can_config.sh

   cd /home/agilex/FluxVla/openpi/RLinf

   /usr/bin/python3 \
     toolkits/realworld_check/configure_piper_mode2_master_offset.py \
     --can can_left --apply --restore-default \
     --i-confirm-follower-disconnected

   /usr/bin/python3 \
     toolkits/realworld_check/configure_piper_mode2_master_offset.py \
     --can can_right --apply --restore-default \
     --i-confirm-follower-disconnected

Verify that both buses emit ``0x15x`` and no longer emit ``0x16x`` before
powering off, reconnecting the followers, and powering all four arms. Native
leader-follower motion without ROS is the final recovery check. Never issue a
role-setting command while a follower remains connected to the shared bus.

For one-time recovery after an unclean isolated session, the guarded
``/rlinf/mode2/align_followers_to_leaders`` service can move followers toward
leaders with bounded interpolation before the offline hardware recovery. It
does not change CAN addresses. Never run this service while an RLT policy is
publishing.
