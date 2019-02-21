from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import sys
sys.path.append('/next/u/kgoel/hmr')
sys.path.append('/next/u/kgoel/hmr/SmoothStream')
import argparse

import cv2
import numpy as np
import zmq

from constants import PORT
from utils import *


from absl import flags

import skimage.io as io
import tensorflow as tf

from src.util import renderer as vis_util
from src.util import image as img_util
from src.util import openpose as op_util
import src.config
from src.RunModel import RunModel
import time

from Streamer import Streamer


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

    def process_stream(self, sess, model, config, streamer=None):
        self.keep_running = True
        frames_processed = 0
        renderer = vis_util.SMPLRenderer(face_path=config.smpl_face_path)
        while self.footage_socket and self.keep_running:
            try:
                if frames_processed == 0:
                    start = time.time()

                payload = self.footage_socket.recv_string()
                frame, id = payload.split("__".encode())
                id = int(id.decode())
                print (id)
                self.current_frame = string_to_image(frame)

                # This shouldn't make any difference since we preprocessed before sending the image
                input_img, proc_param, img = preprocess_image(self.current_frame, config)
                input_img = np.expand_dims(input_img, 0)
                # Predict the joints
                joints, verts, cams, joints3d, theta = model.predict(input_img, get_theta=True)

                # We should send this at some point
                # message = combine_encoded_strings(nparray_to_string(joints), nparray_to_string(verts),
                #                                   nparray_to_string(cams), nparray_to_string(joints3d),
                #                                   nparray_to_string(theta))
                
                skel_img, rend_img = visualize(img, proc_param, joints[0], verts[0], cams[0], renderer)
    
#                 print (joints3d, theta)
                frames_processed += 1
                print ("FPS", frames_processed/float(time.time() - start))
                
                if streamer is not None:
                    streamer.footage_socket.send(image_to_string(skel_img))


            except KeyboardInterrupt:
                break
        print("Streaming Stopped!")


    def stop(self):
        """
        Sets 'keep_running' to False to stop the running loop if running.
        :return: None
        """
        self.keep_running = False

def main(sess, model, config, streamer):
    port = "8880"#PORT

#     parser = argparse.ArgumentParser()
#     parser.add_argument('-p', '--port',
#                         help='The port which you want the Streaming Viewer to use, default'
#                              ' is ' + PORT, required=False)

#     args = parser.parse_args()
#     if args.port:
#         port = args.port

    stream_viewer = StreamViewer(port)
    stream_viewer.process_stream(sess, model, config, streamer)


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
    config(sys.argv)
    # Using pre-trained model, change this to use your own.
    config.load_path = src.config.PRETRAINED_MODEL
    config.batch_size = 1

    sess = tf.Session()
    model = RunModel(config, sess=sess)

    return sess, model, config


def run_image(model, img, config):
    input_img, proc_param, img = preprocess_image(img, config)
    # Add batch dimension: 1 x D x D x 3
    input_img = np.expand_dims(input_img, 0)

    # Theta is the 85D vector holding [camera, pose, shape]
    # where camera is 3D [s, tx, ty]
    # pose is 72D vector holding the rotation of 24 joints of SMPL in axis angle format
    # shape is 10D shape coefficients of SMPL
    joints, verts, cams, joints3d, theta = model.predict(input_img, get_theta=True)

    return joints, verts, cams, joints3d, theta


def visualize(img, proc_param, joints, verts, cam, renderer):
    """
    Renders the result in original image coordinate frame.
    """
    cam_for_render, vert_shifted, joints_orig = vis_util.get_original(
        proc_param, verts, cam, joints, img_size=img.shape[:2])

    # Render results
    skel_img = vis_util.draw_skeleton(img, joints_orig)
    rend_img_overlay = renderer(vert_shifted, cam=cam_for_render, img=img, do_alpha=True)
    
    return skel_img, rend_img_overlay


if __name__ == '__main__':
    sess, model, config = setup()
    print ("Setup is complete")
    streamer = Streamer('128.12.254.132', '8080')
    main(sess, model, config, streamer)




