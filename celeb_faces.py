from __future__ import division

# TODO: kNN klasifikace - kazda trida ma nekolik (desitky) zastupcu, pocita se vzdalenost ke kazdemu a pak se urci kNN

import numpy as np
import cv2
import matplotlib.pyplot as plt

from imutils import paths
import imutils
import pickle

from sklearn.preprocessing import LabelEncoder
from sklearn.datasets.base import Bunch
from sklearn.metrics import classification_report

import skimage.feature as skifea

from tqdm import tqdm
import os
import sys


def detect_face(img, fname='face_result.jpg', scaleFactor=1.1, minNeighbors=8, minSize=60, save_face=False):
    # load the face detector and detect faces in the image
    detector = cv2.CascadeClassifier('cascades/haarcascade_frontalface_default.xml')
    faceRects = detector.detectMultiScale(img, scaleFactor=scaleFactor, minNeighbors=minNeighbors, minSize=(minSize, minSize), flags=cv2.cv.CV_HAAR_SCALE_IMAGE)
    # print "I found %d face(s)" % (len(faceRects))

    # loop over the faces and draw a rectangle around each
    image = img.copy()
    for (x, y, w, h) in faceRects:
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # # show the detected faces
    # cv2.imshow("Faces", image)
    # cv2.waitKey(0)

    if save_face:
        im_fname = fname.split(os.path.sep)[-1]
        out_dir = os.path.join('/'.join(fname.split(os.path.sep)[:-3]), '_marked_faces')
        if not os.path.exists(out_dir):
            os.mkdir(out_dir)
        out_fname = os.path.join(out_dir, im_fname)
        cv2.imwrite(out_fname, image)

    return image, len(faceRects), faceRects


def split_to_tiles(img, columns, rows):
    """
    Split an image into a specified number of tiles.
    Args:
       img (ndarray):  The image to split.
       number_tiles (int):  The number of tiles required.
    Returns:
        Tuple of tiles
    """

    im_h, im_w = img.shape
    tile_w, tile_h = int(np.floor(im_w / columns)), int(np.floor(im_h / rows))

    tiles = []
    for pos_y in range(0, im_h - rows, tile_h): # -rows for rounding error.
        for pos_x in range(0, im_w - columns, tile_w): # as above.
            roi = (pos_x, pos_y, pos_x + tile_w, pos_y + tile_h)
            tile = img[roi[1]:roi[3], roi[0]:roi[2]]
            tiles.append(tile)

    return tuple(tiles)


def non_max_suppression_fast(boxes, overlapThresh):
    # if there are no boxes, return an empty list
    if len(boxes) == 0:
        return []

    # if the bounding boxes integers, convert them to floats --
    # this is important since we'll be doing a bunch of divisions
    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")

    # initialize the list of picked indexes
    pick = []

    # grab the coordinates of the bounding boxes
    x1 = boxes[:,0]
    y1 = boxes[:,1]
    x2 = boxes[:,2]
    y2 = boxes[:,3]

    # compute the area of the bounding boxes and sort the bounding
    # boxes by the bottom-right y-coordinate of the bounding box
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(y2)

    # keep looping while some indexes still remain in the indexes
    # list
    while len(idxs) > 0:
        # grab the last index in the indexes list and add the
        # index value to the list of picked indexes
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)

        # find the largest (x, y) coordinates for the start of
        # the bounding box and the smallest (x, y) coordinates
        # for the end of the bounding box
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])

        # compute the width and height of the bounding box
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)

        # compute the ratio of overlap
        overlap = (w * h) / area[idxs[:last]]

        # delete all indexes from the index list that have
        idxs = np.delete(idxs, np.concatenate(([last],
            np.where(overlap > overlapThresh)[0])))

    # return only the bounding boxes that were picked using the
    # integer data type
    return boxes[pick].astype("int")


def hyperparameters_tuning(img):
    scaleFactor_list = [1.05, 1.1, 1.2]
    minNeighbors_list = [5, 8, 11]
    minSize_list = [60, 80, 100]

    it = len(scaleFactor_list) * len(minNeighbors_list) * len(minSize_list)
    curr_it = 0
    res_file = open('/home/tomas/Dropbox/Data/celeb_dataset/hyperparameters_tuning.txt' 'w')
    for sf in scaleFactor_list:
        for mn in minNeighbors_list:
            for ms in minSize_list:
                missed = 0
                extra = 0
                ok = 0
                n_files = 0
                curr_it += 1
                files = paths.list_images(dataset_path)
                # for fname in tqdm(files):
                for fname in files:
                    n_files += 1
                    image = cv2.imread(fname)
                    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

                    # gray = imutils.resize(gray, width=width)

                    im_face, n_faces, bbox,  = detect_face(gray, fname, scaleFactor=sf, minNeighbors=mn, minSize=ms,
                                                           save_face=save_face)

                    if n_faces == 0:
                        missed += 1
                    elif n_faces > 1:
                        extra += n_faces - 1
                    else:
                        ok += 1

                # print 'total # of images: ', n_files
                print 'it #%i/%i  -----------------------------------------------' % (curr_it, it)
                print 'params: sf=%.2f, mn=%i, ms=%i' % (sf, mn, ms)
                print 'ok: %i/%i, missed: %i, extra: %i' % (ok, n_files, missed, extra)
                print '\n'

                res_file.write('it #%i/%i  -----------------------------------------------' % (curr_it, it))
                res_file.write('params: sf=%.2f, mn=%i, ms=%i' % (sf, mn, ms))
                res_file.write('ok: %i/%i, missed: %i, extra: %i' % (ok, n_files, missed, extra))
                res_file.write('\n')


def detect_batch(dataset_path, save_face=False):
    scaleFactor = 1.1
    minNeighbors = 8
    minSize = 60

    missed = 0
    extra = 0
    ok = 0
    n_files = 0
    files = paths.list_images(dataset_path)
    faces = []
    names = []

    # files = ['/home/tomas/Dropbox/Data/celeb_dataset/natalie_portman/natalie_portman_07.jpg', ]
    for fname in files:
        n_files += 1
        image = cv2.imread(fname)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        im_face, n_faces, bbox = detect_face(gray, fname, scaleFactor=scaleFactor, minNeighbors=minNeighbors,
                                             minSize=minSize, save_face=save_face)
        if n_faces == 0:
            missed += 1
            print 'missed:', fname.split('/')[-1]
        elif n_faces > 1:
            extra += n_faces - 1
            print 'extra:', fname.split('/')[-1]
        else:
            ok += 1
            # imgs.append(im_face)
            bbox = bbox[0]
            # bboxes.append(bbox)
            face = gray[bbox[1]:bbox[1] + bbox[3], bbox[0]:bbox[0] + bbox[2]]
            faces.append(face)
            name = fname.split('/')[-1].split('.')[0]
            names.append(name)


    # print 'total # of images: ', n_files
    print 'ok: %i/%i, missed: %i, extra: %i' % (ok, n_files, missed, extra)
    return faces, names


def train_classifier(dataset_path):
    # (training, testing, names) = load_caltech_faces(args["dataset"], min_faces=21, test_size=0.25)

    faces, names = detect_batch(dataset_path)
    # for f in faces:
    #     f = cv2.resize(face, face_size)

    # create the training and testing bunches
    training = Bunch(name='training', data=faces, target=names)

    # encode the labels, transforming them from strings into integers since OpenCV does
    # not like strings as training data
    le = LabelEncoder()
    le.fit_transform(training.target)

    # train the Local Binary Pattern face recognizer
    print("[INFO] training face recognizer...")
    recognizer = cv2.createLBPHFaceRecognizer(radius=2, neighbors=16, grid_x=8, grid_y=8)
    recognizer.train(training.data, le.transform(training.target))

    return recognizer


def prepare_dataset(dataset_path, face_size=(47, 62)):
    faces, names = detect_batch(dataset_path)
    faces_res = []
    for f in faces:
        f = cv2.resize(f, face_size)
        faces_res.append(f)

    faces_d = {key: val for key, val in zip(names, faces_res)}
    return faces_d


def describe_dataset(faces, numPoints=16, radius=2, grid_x=8, grid_y=8):
    hists = {}
    for name, face in faces.items():
        lbp_hist = describe_im(face, numPoints=numPoints, radius=radius, grid_x=grid_x, grid_y=grid_y)
        # hists.append(lbp_hist)
        hists[name] = lbp_hist
    return hists


def describe_im(img, numPoints=16, radius=2, grid_x=8, grid_y=8):
    tiles = split_to_tiles(img, grid_x, grid_y)
    lbp_hist = []
    for tile in tiles:
        lbp = skifea.local_binary_pattern(tile, numPoints, radius, method='uniform')
        (hist, _) = np.histogram(lbp.ravel(), bins=range(0, numPoints + 3), range=(0, numPoints + 2))

        # optionally normalize the histogram
        eps = 1e-7
        hist = hist.astype("float")
        hist /= (hist.sum() + eps)
        lbp_hist += [x for x in hist]
    return lbp_hist


def run(test_im, dataset_path, lbps_fname, faces_fname, scaleFactor=1.1, minNeighbors=8, minSize=60,
        numPoints=16, radius=2, grid_x=8, grid_y=8):
    if not os.path.exists(faces_fname):
        print 'extracting faces from dataset ...',
        faces = prepare_dataset(dataset_path)
        print 'done'
        print 'saving extracted faces to file ...',
        # np.save(faces_fname, faces)
        with open(faces_fname, 'wb') as handle:
            pickle.dump(faces, handle)
        print 'done'
    else:
        print 'loading faces from dataset ...',
        # faces = np.load(faces_fname)
        with open(faces_fname, 'rb') as handle:
            faces = pickle.load(handle)
        print 'done'

    if not os.path.exists(lbps_fname):
        print 'describing dataset ...',
        hists = describe_dataset(faces)
        print 'done'
        print 'saving description to file ...',
        # np.save(lbps_fname, hists)
        with open(lbps_fname, 'wb') as handle:
            pickle.dump(hists, handle)
        print 'done'
    else:
        print 'loading description from file ...',
        # hists = np.load(lbps_fname)
        with open(lbps_fname, 'rb') as handle:
            hists = pickle.load(handle)
        print 'done'

    im_face, n_faces, bbox = detect_face(test_im, scaleFactor=scaleFactor, minNeighbors=minNeighbors, minSize=minSize)
    if n_faces == 0:
        print 'No face detected.'
    elif n_faces > 1:
        print 'Detected more than 1 face.'
        cv2.imshow('detected face', im_face)
        cv2.waitKey(0)
    else:
        bbox = bbox[0]
        test_face = test_im[bbox[1]:bbox[1] + bbox[3], bbox[0]:bbox[0] + bbox[2]]
        test_face = cv2.resize(test_face, (47, 62))
        lbp_hist = describe_im(test_face, numPoints=numPoints, radius=radius, grid_x=grid_x, grid_y=grid_y)
        lbp_hist = np.array(lbp_hist)

        results = {}
        # loop over items in the dataset
        for (k, features) in hists.items():
            # compute the chi-squared distance between the current features and the query
            # features, then update the dictionary of results
            d = 0.5 * np.sum(((np.array(features) - lbp_hist) ** 2) / (np.array(features) + lbp_hist + 1e-10))
            results[k] = d

        # sort the results
        results = sorted([(v, k) for (k, v) in results.items()])[:5]

        # visualize the results
        plt.figure()
        plt.subplot(231)
        plt.imshow(test_face, 'gray')
        plt.title('test image')
        for (i, (score, name)) in enumerate(results):
            plt.subplot(2, 3, i+2)
            plt.imshow(faces[name], 'gray')
            plt.title("#%d %s: %.4f" % (i + 1, name, score))
        plt.show()

# ---------------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':
    save_face = False
    dataset_path = '/home/tomas/Dropbox/Data/celeb_dataset/celebrities/'
    # detect_batch(dataset_path, save_face=save_face)
    # recognizer_fname = '/home/tomas/Dropbox/Data/celeb_dataset/recognizer_lbp.yaml'
    faces_fname = '/home/tomas/Dropbox/Data/celeb_dataset/dataset_faces.pkl'
    lbps_fname = '/home/tomas/Dropbox/Data/celeb_dataset/dataset_lbp.pkl'

    test_fname = '/home/tomas/Dropbox/Data/celeb_dataset/testing_set/natalie_portman_test1.jpg'
    # test_fname = '/home/tomas/Dropbox/Data/celeb_dataset/celebrities/natalie_portman/natalie_portman_05.jpg'
    test_im = cv2.imread(test_fname, 0)
    run(test_im, dataset_path, lbps_fname, faces_fname)