# -- coding: UTF-8
"""Record a supervised-inference episode that spans the T human takeover.

Recording starts when /teleop/mode first reaches a recording mode (INFERENCE by
default), continues across the T transition, and stops when the operator presses
H, which the supervised keyboard publishes as /teleop/home_request.

Every frame carries the executed action plus the policy and leader actions and
an intervention mask, the schema README_PIPER_FOUR_CAN.md section 8.6 asks for:

    executed_action     /master/joint_left|right   (mux-arbitrated, both phases)
    policy_action       /teleop/policy/joint_left|right
    leader_action       /teleop/leader/joint_left|right
    intervention_mask   /teleop/mode == TAKEOVER

HDF5 is the on-disk staging format because rospy only exists in the ROS Noetic
Python 3.8 interpreter and LeRobot only in the Python 3.11 openpi venv. The
session converts itself by re-invoking hdf5_lerobot.py under that venv.
"""

import argparse
import collections
import os
import subprocess
import sys
import time

import dm_env
import h5py
import numpy as np
import rospy
from sensor_msgs.msg import JointState
from std_msgs.msg import Empty, String

from new_collect_data import KeyboardController, RosOperator

RECORDING_MODES = ("INFERENCE", "TAKEOVER")
DEFAULT_CONVERTER_PYTHON = os.path.join(
    os.environ.get("RLT_ROOT", os.path.expanduser("~/RLT")), "openpi/.venv/bin/python"
)


class TimedSeries:
    """Most recent JointState samples, looked up by frame timestamp."""

    def __init__(self, width=7, maxlen=400):
        self.width = width
        self.samples = collections.deque(maxlen=maxlen)

    def callback(self, message):
        if len(message.position) < self.width:
            return
        self.samples.append(
            (message.header.stamp.to_sec(), [float(v) for v in message.position[:self.width]])
        )

    def at(self, frame_time):
        if not self.samples:
            return [0.0] * self.width
        best = min(self.samples, key=lambda sample: abs(sample[0] - frame_time))
        return list(best[1])


class TakeoverRosOperator(RosOperator):
    """RosOperator plus the supervised-profile action sources and mode labels."""

    def __init__(self, args):
        self.policy = {"left": TimedSeries(), "right": TimedSeries()}
        self.leader = {"left": TimedSeries(), "right": TimedSeries()}
        self.mode = "UNKNOWN"
        self.takeover_phase = "UNKNOWN"
        self.home_requested = False
        super().__init__(args)

    def init_ros(self):
        super().init_ros()
        for side in ("left", "right"):
            rospy.Subscriber(
                "/teleop/policy/joint_{}".format(side),
                JointState,
                self.policy[side].callback,
                queue_size=200,
                tcp_nodelay=True,
            )
            rospy.Subscriber(
                "/teleop/leader/joint_{}".format(side),
                JointState,
                self.leader[side].callback,
                queue_size=200,
                tcp_nodelay=True,
            )
        rospy.Subscriber("/teleop/mode", String, self._mode_callback, queue_size=10)
        rospy.Subscriber(
            "/teleop/takeover_phase", String, self._phase_callback, queue_size=10
        )
        rospy.Subscriber(
            "/teleop/home_request", Empty, self._home_callback, queue_size=1
        )

    def _mode_callback(self, message):
        self.mode = message.data.strip().upper()

    def _phase_callback(self, message):
        self.takeover_phase = message.data.strip().upper()

    def _home_callback(self, _message):
        self.home_requested = True

    def wait_for_recording_mode(self, keyboard, recording_modes):
        print(
            "\033[34mWaiting for /teleop/mode to reach %s. Press q to quit.\033[0m"
            % ", ".join(recording_modes)
        )
        rate = rospy.Rate(20)
        while not rospy.is_shutdown():
            if keyboard is not None and keyboard.read_key() == "quit":
                return "quit"
            if self.mode in recording_modes:
                print("\033[32mmode=%s; recording starts now\033[0m" % self.mode)
                return "start"
            rate.sleep()
        return "quit"

    def record_session(self, keyboard, recording_modes):
        """Record until H, q, or the frame cap. Returns (timesteps, actions, extras, reason)."""
        timesteps = []
        actions = []
        extras = {
            "policy_action": [],
            "leader_action": [],
            "intervention_mask": [],
            "mode": [],
            "takeover_phase": [],
        }
        self.home_requested = False
        count = 0
        rate = rospy.Rate(self.args.frame_rate)
        sync_warned = False
        seen_takeover = False

        while count < self.args.max_timesteps + 1 and not rospy.is_shutdown():
            if self.home_requested:
                print("\n\033[32mH pressed (/teleop/home_request); ending episode.\033[0m")
                return timesteps, actions, extras, "home"
            if keyboard is not None and keyboard.read_key() == "quit":
                print("\nQuit requested by keyboard.")
                return timesteps, actions, extras, "quit"

            result = self.get_frame()
            if not result:
                if not sync_warned:
                    print("syn fail")
                    sync_warned = True
                rate.sleep()
                continue
            sync_warned = False
            count += 1
            (img_front, img_left, img_right, img_front_depth, img_left_depth,
             img_right_depth, puppet_arm_left, puppet_arm_right, master_arm_left,
             master_arm_right, robot_base) = result

            image_dict = {
                self.args.camera_names[0]: img_front,
                self.args.camera_names[1]: img_left,
                self.args.camera_names[2]: img_right,
            }
            obs = collections.OrderedDict()
            obs["images"] = image_dict
            if self.args.use_depth_image:
                obs["images_depth"] = {
                    self.args.camera_names[0]: img_front_depth,
                    self.args.camera_names[1]: img_left_depth,
                    self.args.camera_names[2]: img_right_depth,
                }
            obs["qpos"] = np.concatenate(
                (np.array(puppet_arm_left.position), np.array(puppet_arm_right.position))
            )
            obs["qvel"] = np.concatenate(
                (np.array(puppet_arm_left.velocity), np.array(puppet_arm_right.velocity))
            )
            obs["effort"] = np.concatenate(
                (np.array(puppet_arm_left.effort), np.array(puppet_arm_right.effort))
            )
            if self.args.use_robot_base:
                obs["base_vel"] = [
                    robot_base.twist.twist.linear.x,
                    robot_base.twist.twist.angular.z,
                ]
            else:
                obs["base_vel"] = [0.0, 0.0]

            if count == 1:
                timesteps.append(
                    dm_env.TimeStep(
                        step_type=dm_env.StepType.FIRST,
                        reward=None,
                        discount=None,
                        observation=obs,
                    )
                )
                continue

            timesteps.append(
                dm_env.TimeStep(
                    step_type=dm_env.StepType.MID,
                    reward=None,
                    discount=None,
                    observation=obs,
                )
            )
            actions.append(
                np.concatenate(
                    (np.array(master_arm_left.position), np.array(master_arm_right.position))
                )
            )

            # The mux already timestamps every arbitrated command, so the
            # follower feedback stamp is the frame's reference clock.
            frame_time = puppet_arm_left.header.stamp.to_sec()
            mode = self.mode
            if mode == "TAKEOVER":
                seen_takeover = True
            extras["policy_action"].append(
                np.array(self.policy["left"].at(frame_time) + self.policy["right"].at(frame_time))
            )
            extras["leader_action"].append(
                np.array(self.leader["left"].at(frame_time) + self.leader["right"].at(frame_time))
            )
            extras["intervention_mask"].append(1 if mode == "TAKEOVER" else 0)
            extras["mode"].append(mode)
            extras["takeover_phase"].append(self.takeover_phase)

            if mode not in recording_modes:
                print(
                    "\n\033[33mmode left the recording set (%s); ending episode.\033[0m"
                    % mode
                )
                return timesteps, actions, extras, "mode_exit"

            sys.stdout.write(
                "\rframe %d  mode=%-9s phase=%-18s takeover_frames=%d"
                % (count, mode, self.takeover_phase, sum(extras["intervention_mask"]))
            )
            sys.stdout.flush()
            rate.sleep()

        if not seen_takeover:
            print("\n\033[33mReached the frame cap without any TAKEOVER frame.\033[0m")
        return timesteps, actions, extras, "max"


def save_data(args, timesteps, actions, extras, dataset_path):
    data_size = len(actions)
    data_dict = {
        "/observations/qpos": [],
        "/observations/qvel": [],
        "/observations/effort": [],
        "/action": [],
        "/base_action": [],
    }
    for cam_name in args.camera_names:
        data_dict["/observations/images/{}".format(cam_name)] = []
        if args.use_depth_image:
            data_dict["/observations/images_depth/{}".format(cam_name)] = []

    while actions:
        action = actions.pop(0)
        ts = timesteps.pop(0)
        data_dict["/observations/qpos"].append(ts.observation["qpos"])
        data_dict["/observations/qvel"].append(ts.observation["qvel"])
        data_dict["/observations/effort"].append(ts.observation["effort"])
        data_dict["/action"].append(action)
        data_dict["/base_action"].append(ts.observation["base_vel"])
        for cam_name in args.camera_names:
            data_dict["/observations/images/{}".format(cam_name)].append(
                ts.observation["images"][cam_name]
            )
            if args.use_depth_image:
                data_dict["/observations/images_depth/{}".format(cam_name)].append(
                    ts.observation["images_depth"][cam_name]
                )

    t0 = time.time()
    with h5py.File(dataset_path + ".hdf5", "w", rdcc_nbytes=1024 ** 2 * 2) as root:
        root.attrs["sim"] = False
        root.attrs["compress"] = False
        root.attrs["profile"] = "takeover"
        root.attrs["takeover_frames"] = int(sum(extras["intervention_mask"]))
        obs = root.create_group("observations")
        image = obs.create_group("images")
        for cam_name in args.camera_names:
            image.create_dataset(
                cam_name, (data_size, 480, 640, 3), dtype="uint8", chunks=(1, 480, 640, 3)
            )
        if args.use_depth_image:
            image_depth = obs.create_group("images_depth")
            for cam_name in args.camera_names:
                image_depth.create_dataset(
                    cam_name, (data_size, 480, 640), dtype="uint16", chunks=(1, 480, 640)
                )
        obs.create_dataset("qpos", (data_size, 14))
        obs.create_dataset("qvel", (data_size, 14))
        obs.create_dataset("effort", (data_size, 14))
        root.create_dataset("action", (data_size, 14))
        root.create_dataset("base_action", (data_size, 2))
        for name, array in data_dict.items():
            root[name][...] = array

        root.create_dataset("policy_action", (data_size, 14))
        root["policy_action"][...] = np.asarray(extras["policy_action"][:data_size])
        root.create_dataset("leader_action", (data_size, 14))
        root["leader_action"][...] = np.asarray(extras["leader_action"][:data_size])
        root.create_dataset(
            "intervention_mask",
            data=np.asarray(extras["intervention_mask"][:data_size], dtype=np.uint8),
        )
        string_type = h5py.special_dtype(vlen=str)
        root.create_dataset(
            "mode", data=np.asarray(extras["mode"][:data_size], dtype=object), dtype=string_type
        )
        root.create_dataset(
            "takeover_phase",
            data=np.asarray(extras["takeover_phase"][:data_size], dtype=object),
            dtype=string_type,
        )
    print(
        "\033[32m\nSaved %d frames (%d takeover) in %.1fs: %s\033[0m\n"
        % (
            data_size,
            int(sum(extras["intervention_mask"][:data_size])),
            time.time() - t0,
            dataset_path,
        )
    )


def convert_to_lerobot(args, dataset_dir):
    script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hdf5_lerobot.py")
    command = [
        args.converter_python,
        script,
        "--datasetDir", dataset_dir,
        "--datasetName", args.lerobot_name,
        "--targetDir", args.lerobot_dir,
        "--instruction", args.instruction,
        "--fps", str(args.frame_rate),
        "--includeIntervention",
    ]
    print("\033[34m\nConverting to LeRobot format:\n  %s\033[0m\n" % " ".join(command))
    result = subprocess.run(command)
    if result.returncode != 0:
        print(
            "\033[31mLeRobot conversion failed (exit %d). The HDF5 episodes in %s are "
            "intact; rerun hdf5_lerobot.py manually.\033[0m" % (result.returncode, dataset_dir)
        )
    return result.returncode


def get_arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset_dir", type=str, default="./data")
    parser.add_argument("--task_name", type=str, default="piper_takeover")
    parser.add_argument("--episode_idx", type=int, default=0)
    parser.add_argument("--max_timesteps", type=int, default=3000,
                        help="Safety cap on frames per episode; H normally ends it.")
    parser.add_argument("--camera_names", nargs="+",
                        default=["cam_high", "cam_left_wrist", "cam_right_wrist"])
    parser.add_argument("--img_front_topic", type=str, default="/camera_f/color/image_raw")
    parser.add_argument("--img_left_topic", type=str, default="/camera_l/color/image_raw")
    parser.add_argument("--img_right_topic", type=str, default="/camera_r/color/image_raw")
    parser.add_argument("--img_front_depth_topic", type=str, default="/camera_f/depth/image_raw")
    parser.add_argument("--img_left_depth_topic", type=str, default="/camera_l/depth/image_raw")
    parser.add_argument("--img_right_depth_topic", type=str, default="/camera_r/depth/image_raw")
    # The supervised mux republishes whichever source owns the arm, so this one
    # topic is the executed action during both INFERENCE and TAKEOVER.
    parser.add_argument("--master_arm_left_topic", type=str, default="/master/joint_left")
    parser.add_argument("--master_arm_right_topic", type=str, default="/master/joint_right")
    parser.add_argument("--puppet_arm_left_topic", type=str, default="/puppet/joint_left")
    parser.add_argument("--puppet_arm_right_topic", type=str, default="/puppet/joint_right")
    parser.add_argument("--robot_base_topic", type=str, default="/odom")
    parser.add_argument("--use_robot_base", action="store_true")
    parser.add_argument("--use_depth_image", action="store_true")
    parser.add_argument("--frame_rate", type=int, default=30)
    parser.add_argument("--recording_modes", nargs="+", default=list(RECORDING_MODES),
                        help="Modes that keep the recorder running.")
    parser.add_argument("--require_takeover", action="store_true",
                        help="Discard an episode that contains no TAKEOVER frame.")
    parser.add_argument("--instruction", type=str, default="DEBUG",
                        help="Language instruction stored on every LeRobot frame.")
    parser.add_argument("--lerobot_dir", type=str, default=None,
                        help="LeRobot output directory. Defaults to <dataset_dir>/<task>_lerobot.")
    parser.add_argument("--lerobot_name", type=str, default=None,
                        help="LeRobot repo_id. Defaults to local/<task_name>.")
    parser.add_argument("--converter_python", type=str, default=DEFAULT_CONVERTER_PYTHON,
                        help="Interpreter that has lerobot installed.")
    parser.add_argument("--no_convert", action="store_true",
                        help="Keep HDF5 only; do not run the LeRobot conversion.")
    args = parser.parse_args()
    if args.lerobot_dir is None:
        args.lerobot_dir = os.path.join(args.dataset_dir, args.task_name + "_lerobot")
    if args.lerobot_name is None:
        args.lerobot_name = "local/" + args.task_name
    return args


def main():
    args = get_arguments()
    dataset_dir = os.path.join(args.dataset_dir, args.task_name)
    if not os.path.exists(dataset_dir):
        os.makedirs(dataset_dir)

    ros_operator = TakeoverRosOperator(args)
    recording_modes = tuple(mode.strip().upper() for mode in args.recording_modes)
    episode_idx = args.episode_idx
    saved = 0

    print(
        "\nSupervised takeover recorder.\n"
        "  Recording begins when /teleop/mode enters %s.\n"
        "  Press T on the supervised keyboard to take over; both phases are kept.\n"
        "  Press H to end the episode (H also sends the arms HOME).\n"
        "  Press q here to finish the session.\n" % ", ".join(recording_modes)
    )
    with KeyboardController() as keyboard:
        while not rospy.is_shutdown():
            if ros_operator.wait_for_recording_mode(keyboard, recording_modes) == "quit":
                break
            print(
                "\033[34mRecording episode_%s (cap %s frames).\033[0m"
                % (episode_idx, args.max_timesteps)
            )
            timesteps, actions, extras, reason = ros_operator.record_session(
                keyboard, recording_modes
            )

            takeover_frames = int(sum(extras["intervention_mask"]))
            if not actions:
                print("\033[31mNo frames recorded; episode not saved.\033[0m")
            elif args.require_takeover and takeover_frames == 0:
                print(
                    "\033[31mEpisode had no TAKEOVER frame and --require_takeover is "
                    "set; not saved.\033[0m"
                )
            else:
                save_data(
                    args,
                    timesteps,
                    actions,
                    extras,
                    os.path.join(dataset_dir, "episode_" + str(episode_idx)),
                )
                episode_idx += 1
                saved += 1

            if reason == "quit":
                break

    if saved and not args.no_convert:
        convert_to_lerobot(args, dataset_dir)
    elif saved:
        print("\033[34mSkipped LeRobot conversion (--no_convert). HDF5 in %s\033[0m" % dataset_dir)


if __name__ == "__main__":
    main()
