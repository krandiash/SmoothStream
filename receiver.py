import argparse
import sys

import cv2
import numpy as np
import zmq

from constants import PORT
from utils import string_to_image

import base64
import time
import blosc

sys.path.append('/Users/krandiash/Desktop/live-feedback/')
from openpose_analysis import lightweight_inference, openpose_analysis





class StreamViewer:
    def __init__(self, port):
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
        self.current_data = None
        self.current_id = None
        self.learn_joint_models()

        self.n_dropped_frames = 0

    def receive_payload(self, payload):
        data, frame, id, timestamp = payload.split(self.separator)

        if time.time() - float(timestamp) > 0.5 and self.n_dropped_frames < 10:
            print ("Dropping %s" % id)
            self.n_dropped_frames += 1
            return False

        self.current_id = int(id)
        self.current_data = blosc.unpack_array(data)
        self.current_frame = string_to_image(frame)

        print (self.current_frame.shape)
        print (self.current_id)
        return True

    def learn_joint_models(self):
        self.joint_models = openpose_analysis(pose_name='warrior')

    def do_inference(self):
        self.current_frame = lightweight_inference(self.joint_models, self.current_frame,
                                                   self.current_data, self.current_id, dev_threshold=2.5)

        print (self.current_frame.shape)

    def receive_payload_image_only(self, payload):
        frame, id, timestamp = payload.split(self.separator)

        if time.time() - float(timestamp) > 0.5:
            print ("Dropping %s" % id)
            return False

        id = int(id)
        print (id)

        self.current_frame = string_to_image(frame)

        return True

    def refresh_view(self):
        if not self.no_display:
            cv2.imshow("Stream", self.current_frame)
            cv2.waitKey(1)

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
            try:
                payload = self.footage_socket.recv()

                refresh_view = self.receive_payload(payload)

                if not refresh_view:
                    continue

                self.n_dropped_frames = 0

                self.do_inference()
                self.refresh_view()

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
    parser.add_argument('-p', '--port', help='The port which you want the Streaming Viewer to use', required=True)
    parser.add_argument('-d', '--no_display', help='Don\'t display images', action='store_true')
    parser.add_argument('-t', '--dev_threshold', help='Threshold for difficulty', default=3.0)

    args = parser.parse_args()

    stream_viewer = StreamViewer(args.port)
    stream_viewer.receive_stream(args.no_display)


if __name__ == '__main__':
    main()
