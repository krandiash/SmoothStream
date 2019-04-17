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
import blosc
from utils import *

import skimage.io as io
import time

from Streamer import Streamer
import os
import queue


class StreamViewer:
    def __init__(self, port, webcam_server, send_port, send_port_tiny):
        """
        Binds the computer to a ip address and starts listening for incoming streams.

        :param port: Port which is used for streaming
        """
        context = zmq.Context()
        self.footage_socket = context.socket(zmq.SUB)
        self.footage_socket.bind('tcp://*:' + str(port))
        self.footage_socket.setsockopt_string(zmq.SUBSCRIBE, np.unicode(''))

        context_send = zmq.Context()
        self.footage_socket_send = context_send.socket(zmq.PUB)
        self.footage_socket_send.connect('tcp://' + str(webcam_server) + ':' + str(send_port))

        context_tiny = zmq.Context()
        self.footage_socket_tiny = context_tiny.socket(zmq.PUB)
        self.footage_socket_tiny.connect('tcp://' + str(webcam_server) + ':' + str(send_port_tiny))

        self.current_frame = None
        self.keep_running = True

        self.n_dropped_frames = 0

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
                frame = self.footage_socket.recv_string(flags=zmq.NOBLOCK)
                self.current_frame = string_to_image(frame)

                if display:
                    cv2.imshow("Stream", self.current_frame)
                    cv2.waitKey(1)

            except KeyboardInterrupt:
                cv2.destroyAllWindows()
                break
        print("Streaming Stopped!")

    def process_stream_openpose(self, data_store, openpose_model_store, face=False, hand=False, store=False):

        self.keep_running = True
        frames_processed = 0

        separator = "______".encode()

        store_folder = data_store + "%s/" % (time.strftime("%b_%d_%Y_%H_%M_%S"))
        os.makedirs(store_folder, exist_ok=True)

        # Build out param dict for OpenPose
        params = dict()
        params["model_folder"] = openpose_model_store
        if face:
            params["face"] = True
        if hand:
            params["hand"] = True

        # Starting OpenPose
        opWrapper = op.WrapperPython()
        opWrapper.configure(params)
        opWrapper.start()

        while self.footage_socket and self.keep_running:

            if frames_processed == 0:
                start = time.time()

            try:
                payload = self.footage_socket.recv(flags=zmq.NOBLOCK)
                frame, id, timestamp = payload.split(separator)

                if time.time() - float(timestamp) > 0.3 and self.n_dropped_frames < 10:
                    print ("Skip %s" % id)
                    self.n_dropped_frames += 1
                    self.footage_socket_tiny.send(str(0).encode())
                    continue

                self.n_dropped_frames = 0
                self.footage_socket_tiny.send(str(1).encode())

                timestamp = float(timestamp)
                id = int(id)

                frame = string_to_image(frame)

                # Add in the current frame
                datum = op.Datum()
                datum.cvInputData = frame

                opWrapper.emplaceAndPop([datum])

                if store:
                    cv2.imwrite(store_folder + 'original_%d.jpg' % id, frame)
                    cv2.imwrite(store_folder + 'rendered_%d.jpg' % id, datum.cvOutputData)
                    np.save(store_folder + 'keypoints_%d' % id, datum.poseKeypoints)

                frames_processed += 1
                print("fps:", frames_processed/float(time.time() - start))

                payload = blosc.pack_array(datum.poseKeypoints) + separator + str(id).encode() + \
                          separator + str(timestamp).encode()

                try:
                    self.footage_socket_send.send(payload, flags=zmq.NOBLOCK)
                except zmq.error.Again:
                    pass

            except zmq.error.Again:
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



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-hp', '--hostport', help='The port which you want to receive data on', default=8880)
    parser.add_argument('-s', '--server', help='IP Address of the destination server', required=True)

    parser.add_argument('-sp', '--sendport', help='The port of the destination server', default=8080)
    parser.add_argument('-cp', '--camport', help='The port of the destination server', default=8081)

    parser.add_argument('--face', help='Use face model', action='store_true')
    parser.add_argument('--hand', help='Use hands model', action='store_true')

    parser.add_argument('--store', help='Whether to store the data', action='store_true')
    parser.add_argument('--data_store', help='Where to store the data',
                        default='/next/u/kgoel/pose_estimation/data/')
    parser.add_argument('--openpose_model_store', help='Where to load pose model from',
                        default='/next/u/kgoel/pose_estimation/openpose/models/')

    args = parser.parse_args()

    stream_viewer = StreamViewer(args.hostport, args.server, int(args.sendport), int(args.camport))
    stream_viewer.process_stream_openpose(args.data_store, args.openpose_model_store, args.face, args.hand, args.store)



