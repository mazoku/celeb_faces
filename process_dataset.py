from __future__ import division, print_function

import cv2
from imutils import encodings
from imutils import paths

def process_dir(dataset_path, output):
    files = paths.list_images(dataset_path)

    # for fname in files:
    f = open(output, 'a+')
    total = 0
    # f.write("{}\n".format(encodings.base64_encode_image(face)))
    total += 1

    print("[INFO] wrote {} frames to file".format(total))
    f.close()


if __name__ == '__main__':
    dataset_path =  '/home/tomas/Dropbox/Data/celeb_dataset/celebrities/'

    process_dir(dataset_path, 'out.')