# -- coding: UTF-8
import argparse
import collections
import os
import select
import sys
import termios
import time
import tty
from collections import deque

import cv2
import dm_env
import h5py
import numpy as np
import rospy
from cv_bridge import CvBridge
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Image, JointState


def decode_key_sequence(sequence):
    if sequence in ('q', 'Q'):
        return 'quit'
    if sequence in ('n', 'N', ' ', '\x1b[C', '\x1bOC', '[C', 'OC'):
        return 'next'
    return None


class KeyboardController:
    def __init__(self):
        self.enabled = sys.stdin.isatty()
        self.old_settings = None

    def __enter__(self):
        if self.enabled:
            self.old_settings = termios.tcgetattr(sys.stdin)
            tty.setcbreak(sys.stdin.fileno())
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if self.enabled and self.old_settings is not None:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.old_settings)

    def read_key(self, timeout=0):
        if not self.enabled:
            return None
        if not select.select([sys.stdin], [], [], timeout)[0]:
            return None

        sequence = sys.stdin.read(1)
        if sequence == '\x1b':
            # Arrow keys arrive as escape sequences such as ESC [ C or ESC O C.
            for _ in range(2):
                if select.select([sys.stdin], [], [], 0.1)[0]:
                    sequence += sys.stdin.read(1)
        elif sequence in ('[', 'O') and select.select([sys.stdin], [], [], 0.1)[0]:
            # If ESC was consumed separately, accept the remaining arrow-key suffix.
            sequence += sys.stdin.read(1)
        return decode_key_sequence(sequence)

    def drain_pending_keys(self):
        drained = []
        if not self.enabled:
            return drained
        while select.select([sys.stdin], [], [], 0)[0]:
            drained.append(sys.stdin.read(1))
        return drained


def prepare_next_episode(keyboard):
    if hasattr(keyboard, "drain_pending_keys"):
        return keyboard.drain_pending_keys()
    return []


def wait_for_next_episode(keyboard, next_episode_delay, clock=time):
    if next_episode_delay >= 0:
        print(
            "\033[34m\nReady for next episode. Press Right Arrow/n/Space to start now, press q to quit, "
            "or wait %.1f seconds.\033[0m\n" % next_episode_delay
        )
        deadline = clock.monotonic() + next_episode_delay
    else:
        print(
            "\033[34m\nReady for next episode. Reposition objects, then press Right Arrow/n/Space to start. "
            "Press q to quit.\033[0m\n"
        )
        deadline = None

    while not rospy.is_shutdown():
        timeout = None if deadline is None else max(0, deadline - clock.monotonic())
        key = keyboard.read_key(timeout=timeout)
        if key == 'quit':
            return 'quit'
        if key == 'next':
            return 'next'
        if deadline is not None and clock.monotonic() >= deadline:
            return 'next'
    return 'quit'


# 保存数据函数
def save_data(args, timesteps, actions, dataset_path):
    # 数据字典
    data_size = len(actions)
    data_dict = {
        # 一个是奖励里面的qpos，qvel， effort ,一个是实际发的acition
        '/observations/qpos': [],
        '/observations/qvel': [],
        '/observations/effort': [],
        '/action': [],
        '/base_action': [],
        # '/base_action_t265': [],
    }

    # 相机字典  观察的图像
    for cam_name in args.camera_names:
        data_dict[f'/observations/images/{cam_name}'] = []
        if args.use_depth_image:
            data_dict[f'/observations/images_depth/{cam_name}'] = []

    # len(action): max_timesteps, len(time_steps): max_timesteps + 1
    # 动作长度 遍历动作
    while actions:
        # 循环弹出一个队列
        action = actions.pop(0)   # 动作  当前动作
        ts = timesteps.pop(0)     # 奖励  前一帧

        # 往字典里面添值
        # Timestep返回的qpos，qvel,effort
        data_dict['/observations/qpos'].append(ts.observation['qpos'])
        data_dict['/observations/qvel'].append(ts.observation['qvel'])
        data_dict['/observations/effort'].append(ts.observation['effort'])

        # 实际发的action
        data_dict['/action'].append(action)
        data_dict['/base_action'].append(ts.observation['base_vel'])

        # 相机数据
        # data_dict['/base_action_t265'].append(ts.observation['base_vel_t265'])
        for cam_name in args.camera_names:
            data_dict[f'/observations/images/{cam_name}'].append(ts.observation['images'][cam_name])
            if args.use_depth_image:
                data_dict[f'/observations/images_depth/{cam_name}'].append(ts.observation['images_depth'][cam_name])

    t0 = time.time()
    with h5py.File(dataset_path + '.hdf5', 'w', rdcc_nbytes=1024**2*2) as root:
        # 文本的属性：
        # 1 是否仿真
        # 2 图像是否压缩
        #
        root.attrs['sim'] = False
        root.attrs['compress'] = False

        # 创建一个新的组observations，观测状态组
        # 图像组
        obs = root.create_group('observations')
        image = obs.create_group('images')
        for cam_name in args.camera_names:
            _ = image.create_dataset(cam_name, (data_size, 480, 640, 3), dtype='uint8',
                                         chunks=(1, 480, 640, 3), )
        if args.use_depth_image:
            image_depth = obs.create_group('images_depth')
            for cam_name in args.camera_names:
                _ = image_depth.create_dataset(cam_name, (data_size, 480, 640), dtype='uint16',
                                             chunks=(1, 480, 640), )

        _ = obs.create_dataset('qpos', (data_size, 14))
        _ = obs.create_dataset('qvel', (data_size, 14))
        _ = obs.create_dataset('effort', (data_size, 14))
        _ = root.create_dataset('action', (data_size, 14))
        _ = root.create_dataset('base_action', (data_size, 2))

        # data_dict write into h5py.File
        for name, array in data_dict.items():  
            root[name][...] = array
    print(f'\033[32m\nSaving: {time.time() - t0:.1f} secs. %s \033[0m\n'%dataset_path)


class RosOperator:
    def __init__(self, args):
        self.robot_base_deque = None
        self.puppet_arm_right_deque = None
        self.puppet_arm_left_deque = None
        self.master_arm_right_deque = None
        self.master_arm_left_deque = None
        self.img_front_deque = None
        self.img_right_deque = None
        self.img_left_deque = None
        self.img_front_depth_deque = None
        self.img_right_depth_deque = None
        self.img_left_depth_deque = None
        self.bridge = None
        self.args = args
        self.init()
        self.init_ros()

    def init(self):
        self.bridge = CvBridge()
        self.img_left_deque = deque()
        self.img_right_deque = deque()
        self.img_front_deque = deque()
        self.img_left_depth_deque = deque()
        self.img_right_depth_deque = deque()
        self.img_front_depth_deque = deque()
        self.master_arm_left_deque = deque()
        self.master_arm_right_deque = deque()
        self.puppet_arm_left_deque = deque()
        self.puppet_arm_right_deque = deque()
        self.robot_base_deque = deque()

    def get_frame(self):
        if len(self.img_left_deque) == 0 or len(self.img_right_deque) == 0 or len(self.img_front_deque) == 0 or \
                (self.args.use_depth_image and (len(self.img_left_depth_deque) == 0 or len(self.img_right_depth_deque) == 0 or len(self.img_front_depth_deque) == 0)):
            return False
        if self.args.use_depth_image:
            frame_time = min([self.img_left_deque[-1].header.stamp.to_sec(), self.img_right_deque[-1].header.stamp.to_sec(), self.img_front_deque[-1].header.stamp.to_sec(),
                              self.img_left_depth_deque[-1].header.stamp.to_sec(), self.img_right_depth_deque[-1].header.stamp.to_sec(), self.img_front_depth_deque[-1].header.stamp.to_sec()])
        else:
            frame_time = min([self.img_left_deque[-1].header.stamp.to_sec(), self.img_right_deque[-1].header.stamp.to_sec(), self.img_front_deque[-1].header.stamp.to_sec()])

        if len(self.img_left_deque) == 0 or self.img_left_deque[-1].header.stamp.to_sec() < frame_time:
            return False
        if len(self.img_right_deque) == 0 or self.img_right_deque[-1].header.stamp.to_sec() < frame_time:
            return False
        if len(self.img_front_deque) == 0 or self.img_front_deque[-1].header.stamp.to_sec() < frame_time:
            return False
        if len(self.master_arm_left_deque) == 0 or self.master_arm_left_deque[-1].header.stamp.to_sec() < frame_time:
            return False
        if len(self.master_arm_right_deque) == 0 or self.master_arm_right_deque[-1].header.stamp.to_sec() < frame_time:
            return False
        if len(self.puppet_arm_left_deque) == 0 or self.puppet_arm_left_deque[-1].header.stamp.to_sec() < frame_time:
            return False
        if len(self.puppet_arm_right_deque) == 0 or self.puppet_arm_right_deque[-1].header.stamp.to_sec() < frame_time:
            return False
        if self.args.use_depth_image and (len(self.img_left_depth_deque) == 0 or self.img_left_depth_deque[-1].header.stamp.to_sec() < frame_time):
            return False
        if self.args.use_depth_image and (len(self.img_right_depth_deque) == 0 or self.img_right_depth_deque[-1].header.stamp.to_sec() < frame_time):
            return False
        if self.args.use_depth_image and (len(self.img_front_depth_deque) == 0 or self.img_front_depth_deque[-1].header.stamp.to_sec() < frame_time):
            return False
        if self.args.use_robot_base and (len(self.robot_base_deque) == 0 or self.robot_base_deque[-1].header.stamp.to_sec() < frame_time):
            return False

        while self.img_left_deque[0].header.stamp.to_sec() < frame_time:
            self.img_left_deque.popleft()
        img_left = self.bridge.imgmsg_to_cv2(self.img_left_deque.popleft(), 'passthrough')
        # print("img_left:", img_left.shape)

        while self.img_right_deque[0].header.stamp.to_sec() < frame_time:
            self.img_right_deque.popleft()
        img_right = self.bridge.imgmsg_to_cv2(self.img_right_deque.popleft(), 'passthrough')

        while self.img_front_deque[0].header.stamp.to_sec() < frame_time:
            self.img_front_deque.popleft()
        img_front = self.bridge.imgmsg_to_cv2(self.img_front_deque.popleft(), 'passthrough')

        while self.master_arm_left_deque[0].header.stamp.to_sec() < frame_time:
            self.master_arm_left_deque.popleft()
        master_arm_left = self.master_arm_left_deque.popleft()

        while self.master_arm_right_deque[0].header.stamp.to_sec() < frame_time:
            self.master_arm_right_deque.popleft()
        master_arm_right = self.master_arm_right_deque.popleft()

        while self.puppet_arm_left_deque[0].header.stamp.to_sec() < frame_time:
            self.puppet_arm_left_deque.popleft()
        puppet_arm_left = self.puppet_arm_left_deque.popleft()

        while self.puppet_arm_right_deque[0].header.stamp.to_sec() < frame_time:
            self.puppet_arm_right_deque.popleft()
        puppet_arm_right = self.puppet_arm_right_deque.popleft()

        img_left_depth = None
        if self.args.use_depth_image:
            while self.img_left_depth_deque[0].header.stamp.to_sec() < frame_time:
                self.img_left_depth_deque.popleft()
            img_left_depth = self.bridge.imgmsg_to_cv2(self.img_left_depth_deque.popleft(), 'passthrough')
            top, bottom, left, right = 40, 40, 0, 0
            img_left_depth = cv2.copyMakeBorder(img_left_depth, top, bottom, left, right, cv2.BORDER_CONSTANT, value=0)

        img_right_depth = None
        if self.args.use_depth_image:
            while self.img_right_depth_deque[0].header.stamp.to_sec() < frame_time:
                self.img_right_depth_deque.popleft()
            img_right_depth = self.bridge.imgmsg_to_cv2(self.img_right_depth_deque.popleft(), 'passthrough')
            top, bottom, left, right = 40, 40, 0, 0
            img_right_depth = cv2.copyMakeBorder(img_right_depth, top, bottom, left, right, cv2.BORDER_CONSTANT, value=0)

        img_front_depth = None
        if self.args.use_depth_image:
            while self.img_front_depth_deque[0].header.stamp.to_sec() < frame_time:
                self.img_front_depth_deque.popleft()
            img_front_depth = self.bridge.imgmsg_to_cv2(self.img_front_depth_deque.popleft(), 'passthrough')
            top, bottom, left, right = 40, 40, 0, 0
            img_front_depth = cv2.copyMakeBorder(img_front_depth, top, bottom, left, right, cv2.BORDER_CONSTANT, value=0)

        robot_base = None
        if self.args.use_robot_base:
            while self.robot_base_deque[0].header.stamp.to_sec() < frame_time:
                self.robot_base_deque.popleft()
            robot_base = self.robot_base_deque.popleft()
        return (img_front, img_left, img_right, img_front_depth, img_left_depth, img_right_depth,
                puppet_arm_left, puppet_arm_right, master_arm_left, master_arm_right, robot_base)

    def img_left_callback(self, msg):
        if len(self.img_left_deque) >= 2000:
            self.img_left_deque.popleft()
        self.img_left_deque.append(msg)

    def img_right_callback(self, msg):
        if len(self.img_right_deque) >= 2000:
            self.img_right_deque.popleft()
        self.img_right_deque.append(msg)

    def img_front_callback(self, msg):
        if len(self.img_front_deque) >= 2000:
            self.img_front_deque.popleft()
        self.img_front_deque.append(msg)

    def img_left_depth_callback(self, msg):
        if len(self.img_left_depth_deque) >= 2000:
            self.img_left_depth_deque.popleft()
        self.img_left_depth_deque.append(msg)

    def img_right_depth_callback(self, msg):
        if len(self.img_right_depth_deque) >= 2000:
            self.img_right_depth_deque.popleft()
        self.img_right_depth_deque.append(msg)

    def img_front_depth_callback(self, msg):
        if len(self.img_front_depth_deque) >= 2000:
            self.img_front_depth_deque.popleft()
        self.img_front_depth_deque.append(msg)

    def master_arm_left_callback(self, msg):
        if len(self.master_arm_left_deque) >= 2000:
            self.master_arm_left_deque.popleft()
        self.master_arm_left_deque.append(msg)

    def master_arm_right_callback(self, msg):
        if len(self.master_arm_right_deque) >= 2000:
            self.master_arm_right_deque.popleft()
        self.master_arm_right_deque.append(msg)

    def puppet_arm_left_callback(self, msg):
        if len(self.puppet_arm_left_deque) >= 2000:
            self.puppet_arm_left_deque.popleft()
        self.puppet_arm_left_deque.append(msg)

    def puppet_arm_right_callback(self, msg):
        if len(self.puppet_arm_right_deque) >= 2000:
            self.puppet_arm_right_deque.popleft()
        self.puppet_arm_right_deque.append(msg)

    def robot_base_callback(self, msg):
        if len(self.robot_base_deque) >= 2000:
            self.robot_base_deque.popleft()
        self.robot_base_deque.append(msg)

    def init_ros(self):
        rospy.init_node('record_episodes', anonymous=True)
        rospy.Subscriber(self.args.img_left_topic, Image, self.img_left_callback, queue_size=1000, tcp_nodelay=True)
        rospy.Subscriber(self.args.img_right_topic, Image, self.img_right_callback, queue_size=1000, tcp_nodelay=True)
        rospy.Subscriber(self.args.img_front_topic, Image, self.img_front_callback, queue_size=1000, tcp_nodelay=True)
        if self.args.use_depth_image:
            rospy.Subscriber(self.args.img_left_depth_topic, Image, self.img_left_depth_callback, queue_size=1000, tcp_nodelay=True)
            rospy.Subscriber(self.args.img_right_depth_topic, Image, self.img_right_depth_callback, queue_size=1000, tcp_nodelay=True)
            rospy.Subscriber(self.args.img_front_depth_topic, Image, self.img_front_depth_callback, queue_size=1000, tcp_nodelay=True)
        
        rospy.Subscriber(self.args.master_arm_left_topic, JointState, self.master_arm_left_callback, queue_size=1000, tcp_nodelay=True)
        rospy.Subscriber(self.args.master_arm_right_topic, JointState, self.master_arm_right_callback, queue_size=1000, tcp_nodelay=True)
        rospy.Subscriber(self.args.puppet_arm_left_topic, JointState, self.puppet_arm_left_callback, queue_size=1000, tcp_nodelay=True)
        rospy.Subscriber(self.args.puppet_arm_right_topic, JointState, self.puppet_arm_right_callback, queue_size=1000, tcp_nodelay=True)
        rospy.Subscriber(self.args.robot_base_topic, Odometry, self.robot_base_callback, queue_size=1000, tcp_nodelay=True)

    def process(self, keyboard=None, episode_idx=None):
        timesteps = []
        actions = []
        # 图像数据
        image = np.random.randint(0, 255, size=(480, 640, 3), dtype=np.uint8)
        image_dict = {}
        for cam_name in self.args.camera_names:
            image_dict[cam_name] = image
        count = 0
        
        # input_key = input("please input s:")
        # while input_key != 's' and not rospy.is_shutdown():
        #     input_key = input("please input s:")

        rate = rospy.Rate(self.args.frame_rate)
        print_flag = True

        while (count < self.args.max_timesteps + 1) and not rospy.is_shutdown():
            key = keyboard.read_key() if keyboard is not None else None
            if key == 'quit':
                print("\nQuit requested by keyboard.")
                return timesteps, actions, 'quit'
            if key == 'next':
                if actions:
                    print("\nNext episode requested by keyboard.")
                    return timesteps, actions, 'next'
                print("\nNo action frames recorded yet; right arrow ignored.")

            # 2 收集数据
            result = self.get_frame()
            if not result:
                if print_flag:
                    print("syn fail")
                    print_flag = False
                rate.sleep()
                continue
            print_flag = True
            count += 1
            (img_front, img_left, img_right, img_front_depth, img_left_depth, img_right_depth,
             puppet_arm_left, puppet_arm_right, master_arm_left, master_arm_right, robot_base) = result
            # 2.1 图像信息
            image_dict = {}
            image_dict[self.args.camera_names[0]] = img_front
            image_dict[self.args.camera_names[1]] = img_left
            image_dict[self.args.camera_names[2]] = img_right

            # 2.2 从臂的信息从臂的状态 机械臂示教模式时 会自动订阅
            obs = collections.OrderedDict()  # 有序的字典
            obs['images'] = image_dict
            if self.args.use_depth_image:
                image_dict_depth = {}
                image_dict_depth[self.args.camera_names[0]] = img_front_depth
                image_dict_depth[self.args.camera_names[1]] = img_left_depth
                image_dict_depth[self.args.camera_names[2]] = img_right_depth
                obs['images_depth'] = image_dict_depth
            obs['qpos'] = np.concatenate((np.array(puppet_arm_left.position), np.array(puppet_arm_right.position)), axis=0)
            obs['qvel'] = np.concatenate((np.array(puppet_arm_left.velocity), np.array(puppet_arm_right.velocity)), axis=0)
            obs['effort'] = np.concatenate((np.array(puppet_arm_left.effort), np.array(puppet_arm_right.effort)), axis=0)
            if self.args.use_robot_base:
                obs['base_vel'] = [robot_base.twist.twist.linear.x, robot_base.twist.twist.angular.z]
            else:
                obs['base_vel'] = [0.0, 0.0]

            # 第一帧 只包含first， fisrt只保存StepType.FIRST
            if count == 1:
                ts = dm_env.TimeStep(
                    step_type=dm_env.StepType.FIRST,
                    reward=None,
                    discount=None,
                    observation=obs)
                timesteps.append(ts)
                continue

            # 时间步
            ts = dm_env.TimeStep(
                step_type=dm_env.StepType.MID,
                reward=None,
                discount=None,
                observation=obs)

            # 主臂保存状态
            action = np.concatenate((np.array(master_arm_left.position), np.array(master_arm_right.position)), axis=0)
            actions.append(action)
            timesteps.append(ts)
            if episode_idx is None:
                print("Frame data: ", count)
            else:
                print("Episode %s frame data: %s" % (episode_idx, count))
            if rospy.is_shutdown():
                exit(-1)
            rate.sleep()

        print("len(timesteps): ", len(timesteps))
        print("len(actions)  : ", len(actions))
        return timesteps, actions, 'max'


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir', action='store', type=str, help='Dataset_dir.',
                        default="./data", required=False)
    parser.add_argument('--task_name', action='store', type=str, help='Task name.',
                        default="aloha_mobile_dummy", required=False)
    parser.add_argument('--episode_idx', action='store', type=int, help='Episode index.',
                        default=0, required=False)
    
    parser.add_argument('--max_timesteps', action='store', type=int, help='Max_timesteps.',
                        default=500, required=False)

    parser.add_argument('--camera_names', action='store', type=str, help='camera_names',
                        default=['cam_high', 'cam_left_wrist', 'cam_right_wrist'], required=False)
    #  topic name of color image
    parser.add_argument('--img_front_topic', action='store', type=str, help='img_front_topic',
                        default='/camera_f/color/image_raw', required=False)
    parser.add_argument('--img_left_topic', action='store', type=str, help='img_left_topic',
                        default='/camera_l/color/image_raw', required=False)
    parser.add_argument('--img_right_topic', action='store', type=str, help='img_right_topic',
                        default='/camera_r/color/image_raw', required=False)
    
    # topic name of depth image
    parser.add_argument('--img_front_depth_topic', action='store', type=str, help='img_front_depth_topic',
                        default='/camera_f/depth/image_raw', required=False)
    parser.add_argument('--img_left_depth_topic', action='store', type=str, help='img_left_depth_topic',
                        default='/camera_l/depth/image_raw', required=False)
    parser.add_argument('--img_right_depth_topic', action='store', type=str, help='img_right_depth_topic',
                        default='/camera_r/depth/image_raw', required=False)
    
    # topic name of arm
    parser.add_argument('--master_arm_left_topic', action='store', type=str, help='master_arm_left_topic',
                        # Four-CAN collection: actions come from the independent
                        # leader reader, never from follower feedback.
                        default='/teleop/leader/joint_left', required=False)
    parser.add_argument('--master_arm_right_topic', action='store', type=str, help='master_arm_right_topic',
                        default='/teleop/leader/joint_right', required=False)
    parser.add_argument('--puppet_arm_left_topic', action='store', type=str, help='puppet_arm_left_topic',
                        default='/puppet/joint_left', required=False)
    parser.add_argument('--puppet_arm_right_topic', action='store', type=str, help='puppet_arm_right_topic',
                        default='/puppet/joint_right', required=False)
    
    # topic name of robot_base
    parser.add_argument('--robot_base_topic', action='store', type=str, help='robot_base_topic',
                        default='/odom', required=False)
    
    parser.add_argument('--use_robot_base', action='store', type=bool, help='use_robot_base',
                        default=False, required=False)
    # collect depth image
    parser.add_argument('--use_depth_image', action='store', type=bool, help='use_depth_image',
                        default=False, required=False)
    
    parser.add_argument('--frame_rate', action='store', type=int, help='frame_rate',
                        default=30, required=False)
    parser.add_argument(
        '--single_episode',
        action='store_true',
        help='Deprecated compatibility option; continuous episode collection is always used.',
    )
    parser.add_argument('--next_episode_delay', action='store', type=float, default=-1.0,
                        help='Seconds to wait before automatically starting the next episode. '
                             'Use -1 to require pressing Right Arrow again.')
    
    args = parser.parse_args()
    return args


def main():
    args = get_arguments()
    if args.single_episode:
        print(
            "\033[33m--single_episode is deprecated and ignored; "
            "press q when you want to save and quit.\033[0m"
        )
    ros_operator = RosOperator(args)
    dataset_dir = os.path.join(args.dataset_dir, args.task_name)
    if not os.path.exists(dataset_dir):
        os.makedirs(dataset_dir)

    episode_idx = args.episode_idx
    print("\nControls: press Right Arrow/n/Space to save current episode; press q to save and quit.\n")
    with KeyboardController() as keyboard:
        while not rospy.is_shutdown():
            print("\033[34m\nRecording episode_%s. Max action frames: %s.\033[0m\n" % (episode_idx, args.max_timesteps))
            timesteps, actions, stop_reason = ros_operator.process(keyboard=keyboard, episode_idx=episode_idx)

            if actions:
                dataset_path = os.path.join(dataset_dir, "episode_" + str(episode_idx))
                save_data(args, timesteps, actions, dataset_path)
                episode_idx += 1
            else:
                print("\033[31m\nNo action frames recorded; episode not saved.\033[0m\n")

            if stop_reason == 'quit':
                break

            wait_result = wait_for_next_episode(keyboard, args.next_episode_delay)
            if wait_result == 'quit':
                break
            prepare_next_episode(keyboard)


if __name__ == '__main__':
    main()

# python collect_data.py --dataset_dir ~/data --max_timesteps 500 --episode_idx 0
