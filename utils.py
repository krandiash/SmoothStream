import io
import numpy as np
import cv2
import base64
import blosc
import sys

def is_raspberry_pi(raise_on_errors=False):
    """Checks if Raspberry PI.

    :return:
    """
    try:
        with io.open('/proc/cpuinfo', 'r') as cpuinfo:
            found = False
            for line in cpuinfo:
                if line.startswith('Hardware'):
                    found = True
                    label, value = line.strip().split(':', 1)
                    value = value.strip()
                    if value not in (
                            'BCM2708',
                            'BCM2709',
                            'BCM2835',
                            'BCM2836'
                    ):
                        if raise_on_errors:
                            raise ValueError(
                                'This system does not appear to be a '
                                'Raspberry Pi.'
                            )
                        else:
                            return False
            if not found:
                if raise_on_errors:
                    raise ValueError(
                        'Unable to determine if this system is a Raspberry Pi.'
                    )
                else:
                    return False
    except IOError:
        if raise_on_errors:
            raise ValueError('Unable to open `/proc/cpuinfo`.')
        else:
            return False

    return True


def preview_image(image, name="window", time=1000):
    cv2.imshow(name, image)
    if cv2.waitKey(time):
        cv2.destroyAllWindows()


def nparray_to_string(arr):
    arr_encoding = base64.b64encode(arr)
    separator_encoding = "__".encode('utf-8')
    shape_encoding = base64.b64encode(np.array(arr.shape))
    return arr_encoding + separator_encoding + shape_encoding


def combine_encoded_strings(*args):
    return "____".encode('utf-8').join(args)


def split_encoded_strings(string):
    return string.split("____".encode('utf-8'))


def string_to_nparray(string):
    arr_encoding, shape_encoding = string.split("__".encode('utf-8'))
    arr = np.fromstring(base64.b64decode(arr_encoding))
    shape = np.fromstring(base64.b64decode(shape_encoding), dtype=int)
    return arr.reshape(shape)


def image_to_string(image):
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90]
    _, buffer = cv2.imencode('.jpg', image, encode_param)
    return blosc.pack_array(buffer)#base64.b64encode(buffer)


def string_to_image(string):
    # img = base64.b64decode(string)
    img = blosc.unpack_array(string)
    print (img)
    # npimg = np.fromstring(img, dtype=np.uint8)
    return cv2.imdecode(img, 1)
