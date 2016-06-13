from __future__ import division

#!/usr/bin/python
# The contents of this file are in the public domain. See LICENSE_FOR_EXAMPLE_PROGRAMS.txt
#
#   This example program shows how to find frontal human faces in an image and
#   estimate their pose.  The pose takes the form of 68 landmarks.  These are
#   points on the face such as the corners of the mouth, along the eyebrows, on
#   the eyes, and so forth.
#
#   This face detector is made using the classic Histogram of Oriented
#   Gradients (HOG) feature combined with a linear classifier, an image pyramid,
#   and sliding window detection scheme.  The pose estimator was created by
#   using dlib's implementation of the paper:
#      One Millisecond Face Alignment with an Ensemble of Regression Trees by
#      Vahid Kazemi and Josephine Sullivan, CVPR 2014
#   and was trained on the iBUG 300-W face landmark dataset.
#
#   Also, note that you can train your own models using dlib's machine learning
#   tools. See train_shape_predictor.py to see an example.
#
#   You can get the shape_predictor_68_face_landmarks.dat file from:
#   http://sourceforge.net/projects/dclib/files/dlib/v18.10/shape_predictor_68_face_landmarks.dat.bz2
#
# COMPILING/INSTALLING THE DLIB PYTHON INTERFACE
#   You can install dlib using the command:
#       pip install dlib
#
#   Alternatively, if you want to compile dlib yourself then go into the dlib
#   root folder and run:
#       python setup.py install
#   or
#       python setup.py install --yes USE_AVX_INSTRUCTIONS
#   if you have a CPU that supports AVX instructions, since this makes some
#   things run faster.
#
#   Compiling dlib should work on any operating system so long as you have
#   CMake and boost-python installed.  On Ubuntu, this can be done easily by
#   running the command:
#       sudo apt-get install libboost-python-dev cmake
#
#   Also note that this example requires scikit-image which can be installed
#   via the command:
#       pip install scikit-image
#   Or downloaded from http://scikit-image.org/download.html.

import sys
import os
import dlib
import glob
from skimage import io
import numpy as np
import cv2

# if len(sys.argv) != 3:
#     print(
#         "Give the path to the trained shape predictor model as the first "
#         "argument and then the directory containing the facial images.\n"
#         "For example, if you are in the python_examples folder then "
#         "execute this program by running:\n"
#         "    ./face_landmark_detection.py shape_predictor_68_face_landmarks.dat ../examples/faces\n"
#         "You can download a trained facial shape predictor from:\n"
#         "    http://sourceforge.net/projects/dclib/files/dlib/v18.10/shape_predictor_68_face_landmarks.dat.bz2")
#     exit()

# predictor_path = sys.argv[1]
# faces_folder_path = sys.argv[2]


def run(predictor_path, faces_folder_path):
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(predictor_path)
    win = dlib.image_window()

    for f in glob.glob(os.path.join(faces_folder_path, "*.jpg")):
        print("Processing file: {}".format(f))
        img = io.imread(f)

        # Ask the detector to find the bounding boxes of each face. The 1 in the
        # second argument indicates that we should upsample the image 1 time. This
        # will make everything bigger and allow us to detect more faces.
        dets = detector(img, 1)
        print("Number of faces detected: {}".format(len(dets)))
        for k, d in enumerate(dets):
            print("Detection {}: Left: {} Top: {} Right: {} Bottom: {}".format(
                k, d.left(), d.top(), d.right(), d.bottom()))
            # Get the landmarks/parts for the face in box d.
            shape = predictor(img, d)

            print("Part 0: {}, Part 1: {} ...".format(shape.part(0),
                                                      shape.part(1)))
            left_eye_idxs = range(42, 48)
            left_eye_pts = np.array([[p.x, p.y] for i, p in enumerate(shape.parts()) if i in left_eye_idxs])
            left_eye = np.mean(left_eye_pts, axis=0)
            right_eye_idxs = range(36, 42)
            right_eye_pts = np.array([[p.x, p.y] for i, p in enumerate(shape.parts()) if i in right_eye_idxs])
            right_eye = np.mean(right_eye_pts, axis=0)

            cv2.circle(img, tuple(left_eye.astype(np.int)), 3, color=(0, 255, 255))
            cv2.circle(img, tuple(right_eye.astype(np.int)), 3, color=(0, 255, 255))
            # Draw the face landmarks on the screen.
            win.add_overlay(shape)


        win.set_image(img)
        win.add_overlay(dets)
        cv2.imwrite(os.path.join(face_folder_path, f.replace('.', '_landmarks.')), img)
        dlib.hit_enter_to_continue()
        win.clear_overlay()


if __name__ == '__main__':
    predictor_path = '/home/tomas/Dropbox/Data/shape_predictor_68_face_landmarks.dat'
    # face_folder_path = '/home/tomas/Dropbox/Data/celeb_dataset/celebrities/natalie_portman'
    face_folder_path = '/home/tomas/Dropbox/Data/celeb_dataset/testing_set'

    run(predictor_path, face_folder_path)