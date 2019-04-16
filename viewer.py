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
    def __init__(self, port):
        """
        Binds the computer to a ip address and starts listening for incoming streams.

        :param port: Port which is used for streaming
        """
        context = zmq.Context()
        self.footage_socket = context.socket(zmq.SUB)
        self.footage_socket.bind('tcp://*:' + str(port))
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
                frame = self.footage_socket.recv_string(flags=zmq.NOBLOCK)
                self.current_frame = string_to_image(frame)

                if display:
                    cv2.imshow("Stream", self.current_frame)
                    cv2.waitKey(1)

            except KeyboardInterrupt:
                cv2.destroyAllWindows()
                break
        print("Streaming Stopped!")

    def process_stream_openpose(self, data_store, openpose_model_store, streamer=None,
                                face=False, hand=False, store=False):

        self.keep_running = True
        frames_processed = 0

        separator = "____".encode()

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
                ready = time.time()
                payload = self.footage_socket.recv()#flags=zmq.NOBLOCK)
                frame, id = payload.split(separator)

                id = int(id)
                # frame = blosc.unpack_array(frame)
                print(id)

                frame = string_to_image(frame)
                # print (self.current_frame.shape)

                # Add in the current frame
                datum = op.Datum()
                datum.cvInputData = frame

                opWrapper.emplaceAndPop([datum])

                if store:
                    print ("Store.")
                    cv2.imwrite(store_folder + 'original_%d.jpg' % id, frame)
                    cv2.imwrite(store_folder + 'rendered_%d.jpg' % id, datum.cvOutputData)
                    np.save(store_folder + 'keypoints_%d' % id, datum.poseKeypoints)

                frames_processed += 1
                print("fps:", frames_processed/float(time.time() - start))

                print(time.time() - ready)

                if streamer is not None:
                    ready = time.time()
                    payload = blosc.pack_array(datum.poseKeypoints) + separator + image_to_string(datum.cvOutputData) \
                              + separator + str(id).encode()
                    # payload = base64.b64encode(datum.poseKeypoints) + separator + image_to_string(datum.cvOutputData) \
                    #           + separator + str(id).encode()
                    print (time.time() - ready)

                    try:
                        ready = time.time()
                        streamer.footage_socket.send(payload)#, flags=zmq.NOBLOCK)
                        print (time.time() - ready)
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


def main(streamer, hostport, face, hand, data_store, openpose_model_store, store):
    stream_viewer = StreamViewer(hostport)
    stream_viewer.process_stream_openpose(data_store, openpose_model_store, streamer, face, hand, store)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-hp', '--hostport', help='The port which you want to receive data on', default=8880)
    parser.add_argument('-s', '--server', help='IP Address of the destination server', required=True)
    parser.add_argument('-sp', '--sendport', help='The port of the destination server', default=8080)

    parser.add_argument('--face', help='Use face model', action='store_true')
    parser.add_argument('--hand', help='Use hands model', action='store_true')

    parser.add_argument('--store', help='Whether to store the data', action='store_true')
    parser.add_argument('--data_store', help='Where to store the data',
                        default='/next/u/kgoel/pose_estimation/data/')
    parser.add_argument('--openpose_model_store', help='Where to load pose model from',
                        default='/next/u/kgoel/pose_estimation/openpose/models/')

    args = parser.parse_args()

    streamer = Streamer(args.server, int(args.sendport))
    main(streamer, args.hostport, args.face, args.hand, args.data_store, args.openpose_model_store, args.store)





