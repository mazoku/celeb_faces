from pyimagesearch.face_recognition import FaceDetector
from pyimagesearch.face_recognition import FaceRecognizer
import train_recognizer
import imutils
import cv2
import matplotlib.pyplot as plt
from imutils import paths

import glob


def read_repre(repre_path):
    # f = open(repre_path, 'r')
    # repres = dict()
    # for l in f:
    #     elems = l.strip().split(' ')
    #     repres[elems[0]] = cv2.imread(elems[1])
    # f.close()
    # return repres
    files = paths.list_images(repre_path)
    repres = {}
    for f in files:
        name = f.split('/')[-1]
        name = name[:name.rfind('.')]
        img = cv2.imread(f)
        repres[name] = img
    return repres


def run(im, dataset_path, base64_path, models_path, face_cascade, repre_path, confidence_t=100):
    repres = read_repre(repre_path)

    img = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    fd = FaceDetector(face_cascade)
    img = imutils.resize(img, width=500)
    faceRects = fd.detect(img, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100))
    if len(faceRects) > 0:
        (x, y, w, h) = max(faceRects, key=lambda b: (b[2] * b[3]))
        face = img[y:y + h, x:x + w].copy(order="C")
    else:
        raise ValueError('No face deteted.')

    train_recognizer.run(dataset_path, base64_path, models_path, face_cascade)

    results = []
    for (i, model_path) in enumerate(glob.glob(models_path + "/*.model")):
        fr = FaceRecognizer.load(model_path)
        fr.setConfidenceThreshold(confidence_t)
        name = model_path.split('/')[-1].replace('.model', '')
        confidence = fr.predict(face)
        results.append((name, confidence))

    results = sorted(results, key=lambda res: res[1])
    print results

    # visualize the results
    k = 5
    plt.figure()
    plt.subplot(231)
    plt.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.title('query')
    for (i, (name, score)) in enumerate(results[:k]):
        plt.subplot(2, 3, i + 2)
        plt.imshow(cv2.cvtColor(repres[name], cv2.COLOR_BGR2RGB))
        plt.title("#%d %s: %.4f" % (i + 1, name, score))
        plt.axis('off')
    plt.show()



# ---------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------
if __name__ == '__main__':
    dataset_path = '/home/tomas/Dropbox/Data/celeb_dataset/celebrities/'
    base64_path = '/home/tomas/Dropbox/Data/celeb_dataset/base64_data/'
    models_path = '/home/tomas/Dropbox/Data/celeb_dataset/models/'
    face_cascade = '/home/tomas/projects/celeb_faces/cascades/haarcascade_frontalface_default.xml'
    repre_path = '/home/tomas/Dropbox/Data/celeb_dataset/representatives/'
    confidence_t = 100

    test_fname = '/home/tomas/Dropbox/Data/celeb_dataset/testing_set/natalie_portman_test1.jpg'
    # test_fname = '/home/tomas/Dropbox/Data/celeb_dataset/celebrities/natalie_portman/natalie_portman_05.jpg'
    test_im = cv2.imread(test_fname)

    run(test_im, dataset_path, base64_path, models_path, face_cascade, repre_path, confidence_t)