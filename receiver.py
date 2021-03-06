import argparse
import sys
import os

import cv2
import numpy as np
import zmq

from constants import PORT
from utils import string_to_image

import base64
import time
import blosc
import pickle

import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt

sys.path.append('/Users/krandiash/Desktop/live-feedback/')
from openpose_analysis import lightweight_inference, openpose_analysis, BODY25_JOINTS


class StreamViewer:

    DIFFICULTY_THRESHOLDS = [6.0, 4.0, 3.0, 2.5]
    DIFFICULTY_LEVELS = {'easy': 0, 'medium': 1, 'hard': 2, 'expert': 3}
    DIFFICULTY_INDICES = {v: k for k, v in DIFFICULTY_LEVELS.items()}

    FEEDBACK_TYPES = {'none': 0, 'pose_only': 1, 'prototype_only': 2, 'pose_with_bad_bones': 3,
                      'pose_and_prototype': 4, 'pose_with_bad_bones_and_prototype': 5}
    FEEDBACK_INDICES = {v: k for k,v in FEEDBACK_TYPES.items()}

    POSES = {'warrior': 0}
    POSE_INDICES = {v: k for k,v in POSES.items()}

    FEEDBACK_WITH_DIFFICULTY = {FEEDBACK_INDICES[3], FEEDBACK_INDICES[5]}

    def __init__(self, port, cam_port, pose, difficulty, mode, feedback_type, subject, study, log_path, logging_delay,
                 staging):

        print("[HERE] Listening on", port, "for pose data.")
        context = zmq.Context()
        self.footage_socket = context.socket(zmq.SUB)
        self.footage_socket.bind('tcp://*:' + str(port))
        self.footage_socket.setsockopt_string(zmq.SUBSCRIBE, np.unicode(''))

        print("[HERE] Listening on", cam_port, "for camera feed.")
        context_camera = zmq.Context()
        self.footage_socket_camera = context_camera.socket(zmq.SUB)
        self.footage_socket_camera.bind('tcp://*:' + str(cam_port))
        self.footage_socket_camera.setsockopt_string(zmq.SUBSCRIBE, np.unicode(''))

        self.current_frame = None
        self.keep_running = True
        self.current_data = None
        self.current_id = 0

        self.n_dropped_frames = 0

        self.all_frames = {}
        self.info = {'total_ndevs': np.inf, 'mean_ndevs': np.inf, 'max_ndevs': np.inf, 'median_ndevs': np.inf,
                     'top3_ndevs': np.inf, 'top5_ndevs': np.inf}
        self.info_history = []
        self.timestamp_history = []

        self.pose_history = []
        self.difficulty_history = []
        self.feedback_mode_history = []
        self.mastery_history = []
        self.staging_history = []

        self.mode = mode
        self.pose = pose
        self.feedback_type = feedback_type
        self.logging_delay = logging_delay
        self.staging = staging

        # Initialize the difficulty and the threshold
        self.difficulty = self.DIFFICULTY_LEVELS[difficulty]
        self.dev_threshold = self.DIFFICULTY_THRESHOLDS[self.DIFFICULTY_LEVELS[difficulty]]

        # Set up an identifier that uniquely identifies this run
        self.run_identifier = ["pilot", "study"][int(study)] + "." + time.strftime("%b_%d_%Y_%H_%M_%S") + "." + subject

        # Set up the path where we're storing the data
        self.store_path = log_path + '/' + self.run_identifier + '/'

        # Make a directory for this run
        os.makedirs(self.store_path)

        # Learn the joint models for the pose we're teaching
        self.learn_joint_models()

        # The last time we logged
        self.last_logged = time.time()
        self.log_counter = 1
        self.max_index_logged = 0

        self.show_difficulty = (self.feedback_type in self.FEEDBACK_WITH_DIFFICULTY)
        self.show_mastery = False

        self.mastery = None
        self.current_mastery = None
        self.current_stage = 1 if self.staging else None
        self.last_stage_upgrade = time.time()

    def update_difficulty(self, difficulty: int):
        self.difficulty = difficulty
        self.dev_threshold = self.DIFFICULTY_THRESHOLDS[difficulty]
        self.difficulty_history.append((difficulty, time.time()))

    def toggle_feedback_type(self):
        self.feedback_type = self.FEEDBACK_INDICES[(self.FEEDBACK_TYPES[self.feedback_type] + 1)
                                                   % len(self.FEEDBACK_TYPES)]
        self.show_difficulty = (self.feedback_type in self.FEEDBACK_WITH_DIFFICULTY)
        self.feedback_mode_history.append((self.feedback_type, time.time()))

    def toggle_pose(self):
        self.pose = self.POSE_INDICES[(self.POSES[self.pose] + 1) % len(self.POSES)]
        self.pose_history.append((self.pose, time.time()))


    def receive_payload_image_only(self, payload):
        frame, id, timestamp = payload.split(self.separator)

        if time.time() - float(timestamp) > 0.5:
            print ("Dropping %s" % id)
            return False

        id = int(id)
        print (id)

        self.current_frame = string_to_image(frame)

        return True

    def receive_payload(self, payload):
        data, id, timestamp = payload.split(self.separator)

        if time.time() - float(timestamp) > 0.5 and self.n_dropped_frames < 10:
            self.n_dropped_frames += 1
            return False

        if self.mode == 0:
            self.current_id = int(id)
        self.current_data = blosc.unpack_array(data) * 1.0
        self.current_timestamp = float(timestamp)

        return True

    def learn_joint_models(self):
        self.joint_models = openpose_analysis(pose_name=self.pose)

        # Analyzing the deviations of existing joints
        joints_and_stds = []
        for k, v in self.joint_models.items():
            print ()
            for kp, vp in v.items():
                print (kp, vp)
                joints_and_stds.append((vp[0][1], kp))

        for std, bone_pair in sorted(joints_and_stds):
            print ("Bone 1:", BODY25_JOINTS[bone_pair[0][0]], BODY25_JOINTS[bone_pair[0][1]],
                   "Bone 2:", BODY25_JOINTS[bone_pair[1][0]], BODY25_JOINTS[bone_pair[1][1]],
                   "Std:", std)

    def do_inference(self):
        self.current_frame, info = lightweight_inference(self.joint_models, self.all_frames[self.current_id],
                                                         self.current_data, self.current_id, self.feedback_type,
                                                         dev_threshold=self.dev_threshold, stage=self.current_stage)
        self.update_info(info)
        self.update_mastery()
        self.update_stage()

    def key_update(self, k, info):
        if not np.isnan(info[k]):
            if info[k] < self.info[k]:
                self.info[k] = info[k]

    def update_info(self, info):
        # Keep track of the entire history of interaction
        self.info_history.append(info)
        # self.timestamp_history.append(self.current_timestamp)

        # Update with the best value
        for k in self.info:
            self.key_update(k, info)

        if (time.time() - self.last_logged) >= self.logging_delay:
            # Store information
            self.plot_performance_criteria()
            # self.plot_history()
            self.store_info_history()
            # Update the last logging time
            self.last_logged = time.time()

    def update_mastery(self, hold_threshold=3, criteria='max_ndevs'):
        # Grab data from the last few seconds
        for i in range(len(self.timestamp_history) - 1, -1, -1):
            if self.timestamp_history[i] < self.current_timestamp - hold_threshold:
                break

        # Not enough data yet to make a judgment
        if i == 0 or self.current_timestamp - max(self.timestamp_history[i], self.last_stage_upgrade) < hold_threshold:
            return False

        info_slice = [e[criteria] for e in self.info_history[i + 1:]]
        worst_in_slice = np.max(info_slice)

        # Check where the worst lies
        for i, difficulty in enumerate(self.DIFFICULTY_THRESHOLDS):
            if worst_in_slice > difficulty:
                break
            if i == len(self.DIFFICULTY_THRESHOLDS) - 1:
                i += 1

        if self.current_mastery is not None:
            if i - 1 > self.current_mastery:
                self.current_mastery = i - 1
                self.mastery_history.append((self.current_mastery, time.time()))
        else:
            if i - 1 >= 0:
                self.current_mastery = i - 1
                self.mastery_history.append((self.current_mastery, time.time()))

        self.show_mastery = True if self.current_mastery else False


    def update_stage(self):
        if not self.staging:
            return

        if self.current_mastery is not None and self.current_mastery >= self.difficulty and self.current_stage < 4:
            self.current_stage = self.current_stage + 1
            self.current_mastery = None
            self.show_mastery = False
            self.last_stage_upgrade = time.time()
            self.staging_history.append((self.current_stage, time.time()))


    def plot_performance_criteria(self, criteria='max_ndevs'):
        data = [e[criteria] for e in self.info_history]

        plt.plot(np.arange(len(data)), data)
        for th in self.DIFFICULTY_THRESHOLDS:
            plt.axhline(th, linestyle='--')
        plt.title(" ".join(criteria.split("_")))
        plt.savefig(self.store_path + f'history_{criteria}.png')
        plt.close()

    def plot_history(self, criteria='max_ndevs'):
        data = [e[criteria] for e in self.info_history]
        start_time = self.timestamp_history[0]

        plt.plot(np.array(self.timestamp_history) - start_time, data)
        for th in self.DIFFICULTY_THRESHOLDS:
            plt.axhline(th, linestyle='--')

        for e in self.mastery_history:
            if e[0] is not None:
                plt.axvline(e[1] - start_time, color='orange', linestyle='--')
                plt.text(e[1] - start_time, 9, self.DIFFICULTY_INDICES[e[0]], rotation=90)

        for e in self.staging_history:
            plt.axvline(e[1] - start_time, color='pink', linestyle='--')
            plt.text(e[1] - start_time, 9, 'Stage ' + str(e[0]), rotation=90)

        plt.ylim([0, 14])
        plt.xlabel('Time (in seconds)')
        plt.title(" ".join(criteria.split("_")))
        plt.savefig(self.store_path + f'history.png')
        plt.close()


    def store_info_history(self):
        with open(self.store_path + f'info_history_{self.log_counter}', 'wb') as f:
            pickle.dump(self.info_history[self.max_index_logged:], f)
        self.max_index_logged = len(self.info_history)
        self.log_counter += 1

    def refresh_view(self):
        if not self.no_display:
            display_frame = self.current_frame[:,::-1]
            if self.show_difficulty:
                display_frame = self.update_difficulty_display(display_frame)

            display_frame = self.update_pose_display(display_frame)

            if self.show_mastery:
                display_frame = self.update_mastery_display(display_frame)

            cv2.imshow("Stream", display_frame)
            cv2.waitKey(1)

    def update_difficulty_display(self, display_frame):
        # Display the difficulty
        display_frame = cv2.putText(img=display_frame, text=self.DIFFICULTY_INDICES[self.difficulty],
                                    org=(0, 70), fontFace=cv2.FONT_HERSHEY_COMPLEX,
                                    fontScale=3, color=[0, 0, 255], thickness=4)
        return display_frame

    def update_pose_display(self, display_frame):
        # Display the pose
        display_frame = cv2.putText(img=display_frame, text=self.pose,
                                    org=(800, 70), fontFace=cv2.FONT_HERSHEY_COMPLEX,
                                    fontScale=3, color=[0, 0, 255], thickness=4)
        return display_frame

    def update_mastery_display(self, display_frame):
        # Display the mastery
        display_frame = cv2.putText(img=display_frame, text=self.DIFFICULTY_INDICES[self.current_mastery],
                                    org=(800, 700), fontFace=cv2.FONT_HERSHEY_COMPLEX,
                                    fontScale=3, color=[0, 255, 0], thickness=4)
        return display_frame


    def receive_camera_payload(self, payload):
        frame, id = payload.split(self.separator)
        id = int(id)
        frame = string_to_image(frame)
        self.all_frames[id] = frame

    def clean_all_frames(self):
        if not self.current_id:
            return
        for key in list(self.all_frames.keys()):
            if key < self.current_id:
                del self.all_frames[key]


    def parse_key_press(self, key_press):
        # Update difficulty by pressing(1, 2, 3, 4)
        if 49 <= key_press <= 52:
            self.update_difficulty(key_press - 49)

        # Toggle pose feedback by pressing 0
        elif key_press == 48:
            self.toggle_feedback_type()

        # Toggle poses by pressing p
        elif key_press == 112:
            self.toggle_pose()

        if key_press != -1:
            print(key_press)


    def receive_stream(self, no_display=False):
        """
        Displays displayed stream in a window if no arguments are passed.
        Keeps updating the 'current_frame' attribute with the most recent frame, this can be accessed using 'self.current_frame'
        :param display: boolean, If False no stream output will be displayed.
        :return: None
        """
        self.keep_running = True
        self.no_display = no_display
        self.separator = "______".encode()

        while self.footage_socket and self.keep_running:

            key_press = cv2.waitKey(1)
            # Do something based on the key press
            self.parse_key_press(key_press)

            try:
                # Receive data from the camera
                try:
                    payload_camera = self.footage_socket_camera.recv(zmq.NOBLOCK)
                    self.receive_camera_payload(payload_camera)
                    self.clean_all_frames()

                except zmq.error.Again:
                    pass

                # Receive data from the server
                try:
                    payload = self.footage_socket.recv(zmq.NOBLOCK)
                    self.receive_payload(payload)
                except zmq.error.Again:
                    if self.mode == 0:
                        continue
                    else:
                        pass

                # if not refresh_view:
                #     continue

                if self.current_id in self.all_frames:
                    self.n_dropped_frames = 0
                    self.do_inference()
                    self.refresh_view()
                    self.current_id += 1

            except zmq.error.Again:
                pass

            except KeyboardInterrupt:
                cv2.destroyAllWindows()
                break
        print("Streaming Stopped!")

    def stop(self):
        """
        Sets 'keep_running' to False to stop the running loop if running.
        :return: None
        """
        self.keep_running = False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--port', help='The port which you want the Streaming Viewer to use', default=8080)
    parser.add_argument('-cp', '--cam_port', help='The port which you want the Streaming Viewer to use', default=8082)

    parser.add_argument('-nd', '--no_display', help='Don\'t display images', action='store_true')
    parser.add_argument('-m', '--mode', help='Mode for running viewer', type=int, choices=[0, 1], default=0)

    parser.add_argument('--pose', help='Pose to give feedback on', choices=['warrior'], default='warrior')
    parser.add_argument('-d', '--difficulty', help='Difficulty of the system',
                        choices=StreamViewer.DIFFICULTY_LEVELS.keys(), default='hard')
    parser.add_argument('-f', '--feedback_type', help='Type of feedback',
                        choices=StreamViewer.FEEDBACK_TYPES.keys(), required=True)

    parser.add_argument('-sub', '--subject', help='Subject identifier', type=str, required=True)
    parser.add_argument('--study', help='Marking if this is a subject in a study', action='store_true')

    parser.add_argument('-l', '--log_path', help='Path to logging directory', type=str, default=os.getcwd())
    parser.add_argument('-ld', '--logging_delay', help='Delay between successive logging operations (in seconds)',
                        type=float, default=5)

    parser.add_argument('-stg', '--staging', help='Curriculum based staging.', action='store_true')

    args = parser.parse_args()

    stream_viewer = StreamViewer(args.port, args.cam_port, args.pose, args.difficulty, args.mode, args.feedback_type,
                                 args.subject, args.study, args.log_path, args.logging_delay, args.staging)
    stream_viewer.receive_stream(args.no_display)


if __name__ == '__main__':
    main()

