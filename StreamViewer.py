from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import argparse

import cv2
import numpy as np
import zmq

from constants import PORT
from utils import string_to_image

import sys
from absl import flags

import skimage.io as io
import tensorflow as tf

from src.util import renderer as vis_util
from src.util import image as img_util
from src.util import openpose as op_util
import src.config
from src.RunModel import RunModel



class StreamViewer:
    def __init__(self, port=PORT):
        """
        Binds the computer to a ip address and starts listening for incoming streams.

        :param port: Port which is used for streaming
        """
        context = zmq.Context()
        self.footage_socket = context.socket(zmq.SUB)
        self.footage_socket.bind('tcp://*:' + port)
        self.footage_socket.setsockopt_string(zmq.SUBSCRIBE, np.unicode(''))
        self.current_frame = None
        self.keep_running = True

    def receive_stream(self, display=True):
        """
        Displays displayed stream in a window if no arguments are passed.
        Keeps updating the 'current_frame' attribute with the most recent frame, this can be accessed using 'self.current_frame'
        :param display: boolean, If False no stream output will be displayed.
        :return: None
        """
        self.keep_running = True
        while self.footage_socket and self.keep_running:
            try:
                frame = self.footage_socket.recv_string()
                self.current_frame = string_to_image(frame)

                if display:
                    cv2.imshow("Stream", self.current_frame)
                    cv2.waitKey(1)

            except KeyboardInterrupt:
                cv2.destroyAllWindows()
                break
        print("Streaming Stopped!")

    def process_stream(self):
        self.keep_running = True
        while self.footage_socket and self.keep_running:
            try:
                frame = self.footage_socket.recv_string()
                self.current_frame = string_to_image(frame)
                print (self.current_frame.shape)



            except KeyboardInterrupt:
                break
        print("Streaming Stopped!")


    def stop(self):
        """
        Sets 'keep_running' to False to stop the running loop if running.
        :return: None
        """
        self.keep_running = False

def main():
    port = PORT

    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--port',
                        help='The port which you want the Streaming Viewer to use, default'
                             ' is ' + PORT, required=False)

    args = parser.parse_args()
    if args.port:
        port = args.port

    stream_viewer = StreamViewer(port)
    stream_viewer.receive_stream()


def preprocess_image(img, config):
    if img.shape[2] == 4:
        img = img[:, :, :3]

    if np.max(img.shape[:2]) != config.img_size:
        print('Resizing so the max image size is %d..' % config.img_size)
        scale = (float(config.img_size) / np.max(img.shape[:2]))
    else:
        scale = 1.
    center = np.round(np.array(img.shape[:2]) / 2).astype(int)
    # image center in (x,y)
    center = center[::-1]

    crop, proc_param = img_util.scale_and_crop(img, scale, center, config.img_size)

    # Normalize image to [-1, 1]
    crop = 2 * ((crop / 255.) - 0.5)

    return crop, proc_param, img


def setup():
    config = flags.FLAGS
    # Using pre-trained model, change this to use your own.
    config.load_path = src.config.PRETRAINED_MODEL
    config.batch_size = 1

    sess = tf.Session()
    model = RunModel(config, sess=sess)

    return sess, model


def run_image(model, img, config):
    input_img, proc_param, img = preprocess_image(img, config)
    # Add batch dimension: 1 x D x D x 3
    input_img = np.expand_dims(input_img, 0)

    # Theta is the 85D vector holding [camera, pose, shape]
    # where camera is 3D [s, tx, ty]
    # pose is 72D vector holding the rotation of 24 joints of SMPL in axis angle format
    # shape is 10D shape coefficients of SMPL
    joints, verts, cams, joints3d, theta = model.predict(input_img, get_theta=True)

    return joints3d, theta


if __name__ == '__main__':
    main()




