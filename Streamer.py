import argparse

import cv2
import zmq
import blosc

from camera.Camera import Camera
from constants import PORT, SERVER_ADDRESS
from utils import image_to_string

import numpy as np
import time
import sys


class Streamer:

    def __init__(self, server_address, send_port, recv_port):
        """
        Tries to connect to the StreamViewer with supplied server_address and creates a socket for future use.

        :param server_address: Address of the computer on which the StreamViewer is running, default is `localhost`
        :param port: Port which will be used for sending the stream
        """

        print("Connecting to", server_address, "at", send_port)
        context = zmq.Context()
        self.footage_socket = context.socket(zmq.PUB)
        self.footage_socket.connect('tcp://' + str(server_address) + ':' + str(send_port))

        print("Listening on", recv_port)
        context_tiny = zmq.Context()
        self.footage_socket_tiny = context_tiny.socket(zmq.SUB)
        self.footage_socket_tiny.bind('tcp://*:' + str(recv_port))
        self.footage_socket_tiny.setsockopt_string(zmq.SUBSCRIBE, np.unicode(''))

        self.keep_running = True
        self.keyframe = None

    def start(self, framerate=30.):
        """
        Starts sending the stream to the Viewer.
        Creates a camera, takes a image frame converts the frame to string and sends the string across the network
        :return: None
        """

        # config = flags.FLAGS

        self.framerates = [framerate/2., framerate]

        print("Streaming Started...")
        camera = Camera()
        camera.start_capture()

        self.keep_running = True

        id = 0
        separator = "______".encode()

        # time.sleep(2)

        start = time.time()

        while self.footage_socket and self.footage_socket_tiny and self.keep_running:
            try:
                try:
                    normal_framerate = int(self.footage_socket_tiny.recv(flags=zmq.NOBLOCK))
                    framerate = self.framerates[normal_framerate]
                except zmq.error.Again:
                    pass

                time.sleep(0.6 / framerate)

                frame = camera.current_frame.read()  # grab the current frame

                # Preprocessing?
                # crop, proc_param, img = preprocess_image(frame)
                # image_as_string = image_to_string(crop)

                image_as_string = image_to_string(frame)  # encode the frame

                self.footage_socket.send(image_as_string + separator + str(id).encode() +
                                         separator + str(round(time.time(), 2)).encode())  # send it

                id += 1

                print ('Frame: %d, framerate: %2.2f fps' % (id, round(id/(time.time() - start), 2)))

            except KeyboardInterrupt:
                cv2.destroyAllWindows()
                break

        print("Streaming Stopped!")
        cv2.destroyAllWindows()

    def stop(self):
        """
        Sets 'keep_running' to False to stop the running loop if running.
        :return: None
        """
        self.keep_running = False

#
# def resize_img(img, scale_factor):
#     new_size = (np.floor(np.array(img.shape[0:2]) * scale_factor)).astype(int)
#     new_img = cv2.resize(img, (new_size[1], new_size[0]))
#     # This is scale factor of [height, width] i.e. [y, x]
#     actual_factor = [
#         new_size[0] / float(img.shape[0]), new_size[1] / float(img.shape[1])
#     ]
#     return new_img, actual_factor
#
#
# def scale_and_crop(image, scale, center, img_size):
#     image_scaled, scale_factors = resize_img(image, scale)
#     # Swap so it's [x, y]
#     scale_factors = [scale_factors[1], scale_factors[0]]
#     center_scaled = np.round(center * scale_factors).astype(np.int)
#
#     margin = int(img_size / 2)
#     image_pad = np.pad(
#         image_scaled, ((margin, ), (margin, ), (0, )), mode='edge')
#     center_pad = center_scaled + margin
#     # figure out starting point
#     start_pt = center_pad - margin
#     end_pt = center_pad + margin
#     # crop:
#     crop = image_pad[start_pt[1]:end_pt[1], start_pt[0]:end_pt[0], :]
#     proc_param = {
#         'scale': scale,
#         'start_pt': start_pt,
#         'end_pt': end_pt,
#         'img_size': img_size
#     }
#
#     return crop, proc_param
#
#
# def preprocess_image(img):
#     if img.shape[2] == 4:
#         img = img[:, :, :3]
#
#     if np.max(img.shape[:2]) != 224:
#         print('Resizing so the max image size is %d..' % 224)
#         scale = (float(224) / np.max(img.shape[:2]))
#     else:
#         scale = 1.
#     center = np.round(np.array(img.shape[:2]) / 2).astype(int)
#     # image center in (x,y)
#     center = center[::-1]
#
#     crop, proc_param = scale_and_crop(img, scale, center, 224)
#
#     # Normalize image to [-1, 1]
#     crop = 2 * ((crop / 255.) - 0.5)
#
#     return crop, proc_param, img


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--server', help='IP Address of the server which you want to connect to', required=True)
    parser.add_argument('-p', '--port', help='The port which you want the Streaming Server to use', required=True)

    parser.add_argument('-rp', '--recv_port', help='The port where we receive useful statistics', default=8081)

    parser.add_argument('-f', '--framerate', help='Framerate at which to broadcast stream', default=15.0)

    args = parser.parse_args()

    streamer = Streamer(args.server, args.port, args.recv_port)
    streamer.start(float(args.framerate))


if __name__ == '__main__':
    main()
