import argparse

import cv2
import numpy as np
import zmq

from constants import PORT
from utils import string_to_image

import base64
import time
import blosc






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
        self.current_data = None

    def receive_payload(self, payload):
        data, frame, id = payload.split(self.separator)
        id = int(id)
        print (id)

        self.current_data = blosc.unpack_array(data) #np.frombuffer(base64.b64decode(data), dtype=np.float32).reshape(-1, 3)
        self.current_frame = blosc.unpack_array(frame)

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
        self.separator = "____".encode()

        while self.footage_socket and self.keep_running:
            try:
                payload = self.footage_socket.recv_string(flags=zmq.NOBLOCK)

                self.receive_payload(payload)

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

    args = parser.parse_args()

    stream_viewer = StreamViewer(args.port)
    stream_viewer.receive_stream(args.no_display)


if __name__ == '__main__':
    main()
