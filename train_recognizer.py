# USAGE
# python train_recognizer.py --selfies output/faces --classifier output/classifier --sample-size 100

# import the necessary packages
from __future__ import print_function

import os
import sys

from pyimagesearch.face_recognition import FaceRecognizer
from pyimagesearch.face_recognition import FaceDetector
import imutils
from imutils import encodings
from imutils import paths
import numpy as np
import argparse
import random
import glob
import cv2

# # construct the argument parse and parse command line arguments
# ap = argparse.ArgumentParser()
# ap.add_argument("-s", "--selfies", required=True, help="path to the selfies directory")
# ap.add_argument("-c", "--classifier", required=True, help="path to the output classifier directory")
# ap.add_argument("-n", "--sample-size", type=int, default=100, help="maximum sample size for each face")
# args = vars(ap.parse_args())
#
# # initialize the face recognizer and the list of labels
# fr = FaceRecognizer(cv2.createLBPHFaceRecognizer(radius=1, neighbors=8, grid_x=8, grid_y=8))
# labels = []
#
# # loop over the input faces for training
# for (i, path) in enumerate(glob.glob(args["selfies"] + "/*.txt")):
# 	# extract the person from the file name,
# 	name = path[path.rfind("/") + 1:].replace(".txt", "")
# 	print("[INFO] training on '{}'".format(name))
#
# 	# load the faces file, sample it, and initialize the list of faces
# 	sample = open(path).read().strip().split("\n")
# 	sample = random.sample(sample, min(len(sample), args["sample_size"]))
# 	faces = []
#
# 	# loop over the faces in the sample
# 	for face in sample:
# 		# decode the face and update the list of faces
# 		faces.append(encodings.base64_decode_image(face))
#
# 	# train the face detector on the faces and update the list of labels
# 	fr.train(faces, np.array([i] * len(faces)))
# 	labels.append(name)
#
# # update the face recognizer to include the face name labels, then write the model to file
# fr.setLabels(labels)
# fr.save(args["classifier"])


def is_empty(dir_path):
    if len(glob.glob(os.path.join(dir_path, '*'))) == 0:
        return True
    else:
        return False


def create_base64_files(dataset_path, base64_path, face_cascade):
    if not os.path.exists(base64_path):
        os.mkdir(base64_path)
    fd = FaceDetector(face_cascade)

    print("[INFO] Extracting faces from dataset...")
    files = paths.list_images(dataset_path)
    data_dict = {}
    for f in files:
        # name = f.split('/')[-2][:f.split('/')[-1].rfind('.')]
        # label = name[:name.rfind('_')]
        label = f.split('/')[-2]

        img = cv2.imread(f, 0)
        img = imutils.resize(img, width=500)

        # faceRects = fd.detect(img, scaleFactor=1.1, minNeighbors=9, minSize=(100, 100))
        faceRects = fd.detect(img, scaleFactor=1.1, minNeighbors=9, minSize=(40, 40))
        if len(faceRects) > 0:
            (x, y, w, h) = max(faceRects, key=lambda b: (b[2] * b[3]))
            face = img[y:y + h, x:x + w].copy(order="C")
        else:
            # cv2.imshow('No face detected', img)
            # cv2.waitKey(0)
            # raise ValueError('No face deteted.')
            continue

        if data_dict.has_key(label):
            data_dict[label].append(face)
        else:
            data_dict[label] = list((face,))

    # for fname in files:
    for k, v in data_dict.items():
        filename = k + '.txt'
        filepath = os.path.join(base64_path, filename)
        f = open(filepath, 'a+')
        total = 0
        for face in v:
            f.write("{}\n".format(encodings.base64_encode_image(face)))
            total += 1

        print("[INFO] wrote {} frames to file {}".format(total, filename))
        f.close()


def train_recognizers(base64_path, models_path, sample_size=10):
    if not os.path.exists(models_path):
        os.mkdir(models_path)
    # faces = paths.list_files(base64_path, validExts=('.txt',))
    # loop over the input faces for training
    for (i, path) in enumerate(glob.glob(base64_path + "/*.txt")):
        fr = FaceRecognizer(cv2.createLBPHFaceRecognizer(radius=1, neighbors=8, grid_x=8, grid_y=8))
        # labels = []

        # extract the person from the file name,
        name = path[path.rfind("/") + 1:].replace(".txt", "")
        print("[INFO] training on '{}'".format(name))

        # load the faces file, sample it, and initialize the list of faces
        sample = open(path).read().strip().split("\n")
        sample = random.sample(sample, min(len(sample), sample_size))
        faces = []

        # loop over the faces in the sample
        for face in sample:
            # decode the face and update the list of faces
            faces.append(encodings.base64_decode_image(face))

        # train the face detector on the faces and update the list of labels
        fr.train(faces, np.array([i] * len(faces)))
        # labels = name

        # update the face recognizer to include the face name labels, then write the model to file
        fr.setLabels((name,))
        model_path = os.path.join(models_path, name + '.model')
        fr.save(model_path)

def run(dataset_path, base64_path, models_path, face_cascade):
    if not os.path.exists(models_path) or is_empty(models_path):
        if not os.path.exists(base64_path) or is_empty(base64_path):
            if not os.path.exists((dataset_path)) or is_empty(dataset_path):
                raise IOError('No data found.')
            else:
                create_base64_files(dataset_path, base64_path, face_cascade)
        train_recognizers(base64_path, models_path)


# ---------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------
if __name__ == '__main__':
    dataset_path = '/home/tomas/Dropbox/Data/celeb_dataset/celebrities/'
    base64_path = '/home/tomas/Dropbox/Data/celeb_dataset/base64_data/'
    models_path = '/home/tomas/Dropbox/Data/celeb_dataset/models/'
    face_cascade = '/home/tomas/projects/celeb_faces/cascades/haarcascade_frontalface_default.xml'
    run(dataset_path, base64_path, models_path, face_cascade)