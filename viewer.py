from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import sys

sys.path.append('/next/u/kgoel/pose_estimation/openpose/build/python')
from openpose import pyopenpose as op
import argparse

import cv2
import numpy as np
import zmq

from constants import PORT
from utils import *

from absl import flags
import skimage.io as io
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

    def process_stream_openpose(self, streamer=None):
        self.keep_running = True
        frames_processed = 0

        params = dict()
        params["model_folder"] = "/next/u/kgoel/pose_estimation/openpose/models/"

        # Starting OpenPose
        opWrapper = op.WrapperPython()
        opWrapper.configure(params)
        opWrapper.start()

        while self.footage_socket and self.keep_running:
            try:
                if frames_processed == 0:
                    start = time.time()

                payload = self.footage_socket.recv_string()
                frame, id = payload.split("__")
                id = int(id)
                print(id)
                self.current_frame = string_to_image(frame)
                print (self.current_frame.shape)
                # Predict the joints
                datum = op.Datum()
                datum.cvInputData = self.current_frame
                opWrapper.emplaceAndPop([datum])

                print (dir(datum))
                print(datum.poseKeypoints)

                frames_processed += 1
                print("FPS", frames_processed/ float(time.time() - start))

                if streamer is not None:
                    streamer.footage_socket.send(image_to_string(datum.cvOutputData))
                    pass

            except KeyboardInterrupt:
                break
        print("Streaming Stopped!")

    def stop(self):
        """
        Sets 'keep_running' to False to stop the running loop if running.
        :return: None
        """
        self.keep_running = False


def main(streamer):
    port = "8880"
    stream_viewer = StreamViewer(port)
    stream_viewer.process_stream_openpose(streamer)

if __name__ == '__main__':
    streamer = Streamer('DN52eo4c.SUNet', '8080')
    main(streamer)





