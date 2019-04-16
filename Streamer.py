import argparse

import cv2
import zmq
import blosc

from camera.Camera import Camera
from constants import PORT, SERVER_ADDRESS
from utils import image_to_string

import numpy as np
import time


class Streamer:

    def __init__(self, server_address=SERVER_ADDRESS, port=PORT):
        """
        Tries to connect to the StreamViewer with supplied server_address and creates a socket for future use.

        :param server_address: Address of the computer on which the StreamViewer is running, default is `localhost`
        :param port: Port which will be used for sending the stream
        """

        print("Connecting to ", server_address, "at", port)
        context = zmq.Context()
        self.footage_socket = context.socket(zmq.PUB)
        self.footage_socket.connect('tcp://' + str(server_address) + ':' + str(port))
        self.keep_running = True

    def start(self, framerate=None):
        """
        Starts sending the stream to the Viewer.
        Creates a camera, takes a image frame converts the frame to string and sends the string across the network
        :return: None
        """

        # config = flags.FLAGS

        print("Streaming Started...")
        camera = Camera()
        camera.start_capture()

        self.keep_running = True

        id = 0
        separator = "____".encode()

        time.sleep(2)

        start = time.time()

        while self.footage_socket and self.keep_running:
            try:
                frame = camera.current_frame.read()  # grab the current frame

                if framerate is not None:
                    time.sleep(0.6/framerate)  # control the frame rate (works best for 15 fps)

                # Preprocessing?
                # crop, proc_param, img = preprocess_image(frame)
                # image_as_string = image_to_string(crop)

                # image_as_string = image_to_string(frame)  # encode the frame

                # Compression?
                # print (len(image_as_string))
                image_as_string = blosc.pack_array(frame)
                # print (len(image_as_string))
                # image_as_string = str(compress(image_as_string)).encode()
                # print (len(image_as_string))
                self.footage_socket.send(image_as_string + separator + str(id).encode())  # send it

                print (id)
                id += 1

                print ('Framerate: %2.2f fps' % round(id/(time.time() - start), 2))

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
    parser.add_argument('-s', '--server',
                        help='IP Address of the server which you want to connect to', required=True)
    parser.add_argument('-p', '--port',
                        help='The port which you want the Streaming Server to use', required=True)
    parser.add_argument('-f', '--framerate',
                        help='Framerate at which to broadcast stream', default=15.0)

    args = parser.parse_args()

    port = args.port
    server_address = args.server
    framerate = float(args.framerate)

    streamer = Streamer(server_address, port)
    streamer.start(framerate)


if __name__ == '__main__':
    main()
