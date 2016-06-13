from __future__ import division

# TODO: zkontrolovat detekovane obliceje
# TODO: optimizer vratit adam

import os
from collections import namedtuple, defaultdict
import gzip
import pickle

import numpy as np
import matplotlib.pyplot as plt
import cv2
from tqdm import tqdm
import tensorflow as tf

from pyimagesearch.face_recognition import FaceDetector
from imutils import paths
import imutils

from keras.datasets import mnist
from keras.models import Sequential
from keras.models import model_from_json
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Convolution2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.constraints import maxnorm
from sklearn.preprocessing import LabelEncoder
from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import StratifiedKFold
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.cross_validation import cross_val_score
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD


def create_data_item(path, fd):
    # data = namedtuple('data item', ('label', 'path', 'face'))
    data_item = dict()
    img = cv2.imread(path, 0)
    faceRects = fd.detect(img, scaleFactor=1.1, minNeighbors=8, minSize=(20, 20))
    if len(faceRects) > 0:
        (x, y, w, h) = max(faceRects, key=lambda b: (b[2] * b[3]))
        face = img[y:y + h, x:x + w].copy(order="C")
    else:
        # print ('No face deteted.')
        return None

    label = path.split(os.path.sep)[-2]
    # item = data(label = label, path = path, face = face)
    data_item['label'] = label
    data_item['path'] = path
    data_item['face'] = face

    return data_item


def process_files(files):
    file_dict = defaultdict(list)
    for f in files:
        name = f.split(os.path.sep)[-2]
        file_dict[name].append(f)
    return file_dict


def data2files(data_dir, face_cascade, res_file_path, label_dict_path):
    # initializing face detector
    fd = FaceDetector(face_cascade)

    dataset = []
    files = paths.list_images(data_dir)
    file_dict = process_files(files)

    for key in file_dict.keys():
        print 'Processing %s ...' % key,
        celeb_files = file_dict[key]
        n_files = len(celeb_files)
        ok = 0
        for f in celeb_files:
            item = create_data_item(f, fd)
            if item is not None:
                dataset.append(item)
                ok += 1
        print 'done, face found: %i/%i (%i%%)' % (ok, n_files, int(100 * ok / n_files))

    print 'Saving data in pickle format...',
    # data_fname = os.path.join(res_dir, 'data.npz')
    data_file = gzip.open(res_file_path, 'wb', compresslevel=1)
    pickle.dump(dataset, data_file)
    # np.save(data_file, np.array(data_cubes_resh))
    data_file.close()
    print 'done'

    print 'Saving label dictionary ...'
    label_dict = {}
    names = file_dict.keys()
    for i, n in enumerate(names):
        label_dict[n] = i
    labels_file = gzip.open(label_dict_path, 'wb', compresslevel=1)
    pickle.dump(label_dict, labels_file)
    labels_file.close()
    print 'done'


def data2TF_records(dataset, label_dict, tfrecords_path):
    # data = OfficialVectorClassification()
    # trIdx = data.sel_idxs[:]
    labels = [x['label'] for x in dataset]
    faces = [x['face'] for x in dataset]
    trIdx = np.arange(len(labels))

    writer = tf.python_io.TFRecordWriter(tfrecords_path)
    np.random.shuffle(trIdx)
    print 'Saving tf records ...',
    # iterate over each example
    # wrap with tqdm for a progress bar
    for example_idx in tqdm(trIdx):
        face = faces[example_idx].flatten()
        label = label_dict[labels[example_idx]]

        # construct the Example proto boject
        example = tf.train.Example(
            # Example contains a Features proto object
            features=tf.train.Features(
                # Features contains a map of string to Feature proto objects
                feature={
                    # A Feature contains one of either a int64_list,
                    # float_list, or bytes_list
                    'label': tf.train.Feature(
                        int64_list=tf.train.Int64List(value=[label])),
                    'image': tf.train.Feature(
                        int64_list=tf.train.Int64List(value=face.astype("int64"))),
                }))
        # use the proto object to serialize the example to a string
        serialized = example.SerializeToString()
        # write the serialized object to disk
        writer.write(serialized)
    print '... done'


def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)


def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)


def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')


def advanced_tf():
    mnist = load_dataset()

    x = tf.placeholder(tf.float32, shape=[None, 784])
    y_ = tf.placeholder(tf.float32, shape=[None, 10])

    # first convolutional layer
    W_conv1 = weight_variable([5, 5, 1, 32])
    b_conv1 = bias_variable([32])

    x_image = tf.reshape(x, [-1, 28, 28, 1])

    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)

    # second convolutional layer
    W_conv2 = weight_variable([5, 5, 32, 64])
    b_conv2 = bias_variable([64])

    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)

    # densely connected layer
    W_fc1 = weight_variable([7 * 7 * 64, 1024])
    b_fc1 = bias_variable([1024])

    h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

    # dropout
    keep_prob = tf.placeholder(tf.float32)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    # readout layer
    W_fc2 = weight_variable([1024, 10])
    b_fc2 = bias_variable([10])

    y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

    # training and evaluating
    cross_entropy = -tf.reduce_sum(y_ * tf.log(y_conv))
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

    correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # init = tf.initialize_all_variables()
    # sess = tf.Session()
    # sess.run(init)
    sess = tf.InteractiveSession()
    sess.run(tf.initialize_all_variables())
    for i in range(20000):
        batch = mnist.train.next_batch(50)
        if i%100 == 0:
            train_accuracy = accuracy.eval(feed_dict={x:batch[0], y_: batch[1], keep_prob: 1.0})
            print("step %d, training accuracy %g" % (i, train_accuracy))

        sess.run(train_step, feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})
        # train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

    print("test accuracy %g" % accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))


def detect_face(im, face_cascade):
    img = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    fd = FaceDetector(face_cascade)
    img = imutils.resize(img, width=500)
    faceRects = fd.detect(img, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100))
    if len(faceRects) > 0:
        (x, y, w, h) = max(faceRects, key=lambda b: (b[2] * b[3]))
        face = img[y:y + h, x:x + w].copy(order="C")
    else:
        raise ValueError('No face deteted.')
    return face


def create_model(num_classes=10, epochs=10, face_size=(40, 40)):
    # create model
    # model = Sequential()
    # model.add(Convolution2D(32, 5, 5, border_mode='valid', input_shape=(1, face_size[0], face_size[1]), activation='relu'))
    # model.add(MaxPooling2D(pool_size=(2, 2)))
    # model.add(Dropout(0.2))
    # model.add(Flatten())
    # model.add(Dense(128, activation='relu'))
    # model.add(Dense(num_classes, activation='softmax'))

    # simple model ---------------------
    model = Sequential()
    model.add(Convolution2D(32, 3, 3, input_shape=(1, face_size[0], face_size[1]), border_mode='same', activation='relu',
                            W_constraint=maxnorm(3)))
    model.add(Dropout(0.2))
    model.add(Convolution2D(32, 3, 3, activation='relu', border_mode='same', W_constraint=maxnorm(3)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(512, activation='relu', W_constraint=maxnorm(3)))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))

    # deeper model -------------------
    # model = Sequential()
    # model.add(Convolution2D(32, 3, 3, input_shape=(1, face_size[0], face_size[1]), activation='relu', border_mode='same'))
    # model.add(Dropout(0.2))
    # model.add(Convolution2D(32, 3, 3, activation='relu', border_mode='same'))
    # model.add(MaxPooling2D(pool_size=(2, 2)))
    # model.add(Convolution2D(64, 3, 3, activation='relu', border_mode='same'))
    # model.add(Dropout(0.2))
    # model.add(Convolution2D(64, 3, 3, activation='relu', border_mode='same'))
    # model.add(MaxPooling2D(pool_size=(2, 2)))
    # model.add(Convolution2D(128, 3, 3, activation='relu', border_mode='same'))
    # model.add(Dropout(0.2))
    # model.add(Convolution2D(128, 3, 3, activation='relu', border_mode='same'))
    # model.add(MaxPooling2D(pool_size=(2, 2)))
    # model.add(Flatten())
    # model.add(Dropout(0.2))
    # model.add(Dense(1024, activation='relu', W_constraint=maxnorm(3)))
    # model.add(Dropout(0.2))
    # model.add(Dense(512, activation='relu', W_constraint=maxnorm(3)))
    # model.add(Dropout(0.2))
    # model.add(Dense(num_classes, activation='softmax'))

    # LeNet ------------------------------
    # model = Sequential()
    # model.add(Convolution2D(32, 3, 3, input_shape=(1, face_size[0], face_size[1]), activation='relu', border_mode='same'))
    # model.add(Dropout(0.2))
    # model.add(MaxPooling2D(pool_size=(2, 2)))
    # model.add(Convolution2D(16, 3, 3, activation='relu', border_mode='same'))
    # model.add(Dropout(0.2))
    # model.add(MaxPooling2D(pool_size=(2, 2)))
    # model.add(Flatten())
    # # model.add(Dense(36, activation='relu', W_constraint=maxnorm(3)))
    # # model.add(Dropout(0.2))
    # model.add(Dense(num_classes, activation='softmax'))

    # TODO: bez dense na konci

    # learning rate decay
    learning_rate = 0.1
    decay_rate = learning_rate / epochs
    momentum = 0.8
    sgd = SGD(lr=learning_rate, momentum=momentum, decay=decay_rate, nesterov=False)

    # Compile model
    # model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    return model


def train_wrapped(data_dir, res_file_path, face_cascade, model_topol_path, model_weights_path, face_size=(40, 40)):
    # if not os.path.exists(tfrecords_path):
    #     if not os.path.exists(res_file_path) or not os.path.exists(label_dict_path):
    #         data2files(data_dir, face_cascade, res_file_path, label_dict_path)
    #
    #     data_file = gzip.open(res_file_path, 'rb', compresslevel=1)
    #     dataset = pickle.load(data_file)
    #     data_file.close()
    #
    #     label_file = gzip.open(label_dict_path, 'rb', compresslevel=1)
    #     label_dict = pickle.load(label_file)
    #     label_file.close()
    #
    #     data2TF_records(dataset, label_dict, tfrecords_path)

    if not os.path.exists(res_file_path) or not os.path.exists(label_dict_path):
        data2files(data_dir, face_cascade, res_file_path, label_dict_path)

    data_file = gzip.open(res_file_path, 'rb', compresslevel=1)
    dataset = pickle.load(data_file)
    data_file.close()

    # label_file = gzip.open(label_dict_path, 'rb', compresslevel=1)
    # label_dict = pickle.load(label_file)
    # label_file.close()

    # converting data to format adequate for cnn
    X = np.dstack([cv2.resize(x['face'], dsize=face_size) for x in dataset])
    # change from (40, 40, n_faces) to (n_faces, 40, 40)
    X = np.swapaxes(np.swapaxes(X, 2, 0), 1, 2)
    Y = [x['label'] for x in dataset]

    # encode class values as integers
    encoder = LabelEncoder()
    encoder.fit(Y)
    encoded_Y = encoder.transform(Y)
    # convert integers to dummy variables (i.e. one hot encoded)
    dummy_y = np_utils.to_categorical(encoded_Y).astype(np.int)

    # X_train, X_test, y_train, y_test = train_test_split(X, dummy_y, test_size=0.33, random_state=seed)
    kfold = StratifiedKFold(y=Y, n_folds=6, shuffle=True, random_state=seed)

    # reshape to be [samples][channels][width][height]
    # X_train = X_train.reshape(X_train.shape[0], 1, face_size[0], face_size[1]).astype('float32')
    # X_train = X_train.reshape(X_train.shape[0], 1, face_size[0], face_size[1]).astype('float32')
    X = X.reshape(X.shape[0], 1, face_size[0], face_size[1]).astype('float32')

    # normalize inputs from 0-255 to 0-1
    # X_train /= 255
    # X_test /= 255
    X /= 255
    # one hot encode outputs
    # y_train = np_utils.to_categorical(y_train)
    # y_test = np_utils.to_categorical(y_test)
    # num_classes = y_train.shape[1]
    dummy_y = np_utils.to_categorical(dummy_y)
    num_classes = dummy_y.shape[1]

    # build the model
    # model = create_model(num_classes, face_size)
    # create model
    model = KerasClassifier(build_fn=create_model, num_classes=num_classes, face_size=face_size, nb_epoch=5, batch_size=10)

    # serialize model to JSON
    # model_topology_path = os.path.join(model_path, 'model_topology.json')
    # model_json = model.to_json()
    # with open(model_topology_path, 'w') as json_file:
    #     json_file.write(model_json)

    # checkpointing
    # model_weights_path = os.path.join(model_path, 'model_weights_best.h5')
    # checkpoint = ModelCheckpoint(model_weights_path, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    # callbacks_list = [checkpoint]

    # # Fit the model
    # model.fit(X_train, y_train, validation_data=(X_test, y_test), nb_epoch=5, batch_size=5, callbacks=callbacks_list,
    #           verbose=2)
    # # Final evaluation of the model
    # scores = model.evaluate(X_test, y_test, verbose=0)
    # print("Baseline Error: %.2f%%" % (100 - scores[1] * 100))

    results = cross_val_score(model, X, dummy_y, cv=kfold)
    print(results.mean())

    return model


def train(data_dir, res_file_path, face_cascade, model_topol_path, model_weights_path, face_size=(40, 40), n_epochs=20):
    if not os.path.exists(res_file_path) or not os.path.exists(label_dict_path):
        data2files(data_dir, face_cascade, res_file_path, label_dict_path)

    data_file = gzip.open(res_file_path, 'rb', compresslevel=1)
    dataset = pickle.load(data_file)
    data_file.close()

    # converting data to format adequate for cnn
    X = np.dstack([cv2.resize(x['face'], dsize=face_size) for x in dataset])
    # change from (width, height, n_faces) to (n_faces, width, height)
    X = np.swapaxes(np.swapaxes(X, 2, 0), 1, 2)
    Y = [x['label'] for x in dataset]

    # encode class values as integers
    encoder = LabelEncoder()
    encoder.fit(Y)
    encoded_Y = encoder.transform(Y)

    # train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, encoded_Y, test_size=0.33, random_state=seed, stratify=encoded_Y)
    y_train = np_utils.to_categorical(y_train).astype(np.int)
    y_test = np_utils.to_categorical(y_test).astype(np.int)
    num_classes = y_train.shape[1]

    # reshaping data
    X_train = X_train.reshape(X_train.shape[0], 1, face_size[0], face_size[1]).astype('float32')
    X_test = X_test.reshape(X_test.shape[0], 1, face_size[0], face_size[1]).astype('float32')

    # normalize inputs from 0-255 to 0-1
    X_train /= 255
    X_test /= 255

    # build the model
    model = create_model(num_classes, n_epochs, face_size)

    # serialize model to JSON
    model_json = model.to_json()
    with open(model_topol_path, 'w') as json_file:
        json_file.write(model_json)

    datagen = ImageDataGenerator(featurewise_center=False, featurewise_std_normalization=False, zca_whitening=True,
                                 horizontal_flip=True)
    # fit parameters from data
    datagen.fit(X_train)

    # checkpointing
    checkpoint = ModelCheckpoint(model_weights_path, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    stop = EarlyStopping(monitor='val_loss', patience=20, verbose=0, mode='min')
    callbacks_list = [checkpoint, stop]

    # Fit the model
    # model.fit(X_train, y_train, validation_data=(X_test, y_test), nb_epoch=10, batch_size=20, verbose=2)
    history = model.fit_generator(datagen.flow(X_train, y_train, batch_size=25), validation_data=(X_test, y_test),
                                  samples_per_epoch=len(y_train), nb_epoch=n_epochs, callbacks=callbacks_list)

    # Final evaluation of the model
    scores = model.evaluate(X_test, y_test, verbose=0)
    print('Error: %.2f%%' % (100 - scores[1] * 100))

    # summarize history for accuracy
    plt.figure()
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')

    # summarize history for loss
    plt.figure()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper right')

    return model


def run(data_dir, res_file_path, face_cascade, model_topol_path, model_weights_path, test_im, face_size=(30, 50), n_epochs=20):
    query = detect_face(test_im, face_cascade)
    query= cv2.resize(query, dsize=face_size)
    query = query.reshape(1, 1, face_size[0], face_size[1]).astype('float32')

    model = train(data_dir, res_file_path, face_cascade, model_topol_path, model_weights_path, face_size=face_size, n_epochs=n_epochs)
    preds = model.predict(query)
    print preds
    # plt.show()


def load_model(model_topol_path, model_weights_path):
    print 'Loading model from disk ...',
    json_file = open(model_topol_path, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    # load weights into new model
    model.load_weights(model_weights_path)
    print 'done'

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model


# --------------------------------------------------------------------------------------------------
if __name__ == '__main__':
    data_dir = '/home/tomas/Dropbox/Data/celeb_dataset/celebrities'
    res_file_path = '/home/tomas/Dropbox/Data/celeb_dataset/celebrities.pklz'
    label_dict_path = '/home/tomas/Dropbox/Data/celeb_dataset/label_dict.pklz'
    face_cascade = '/home/tomas/projects/celeb_faces/cascades/haarcascade_frontalface_default.xml'
    tfrecords_path = '/home/tomas/Dropbox/Data/celeb_dataset/celebrities.tfrecords'
    model_topol_path = '/home/tomas/Dropbox/Data/celeb_dataset/model_topology.json'
    model_weights_path = '/home/tomas/Dropbox/Data/celeb_dataset/model_weights.h5'

    test_fname = '/home/tomas/Dropbox/Data/celeb_dataset/testing_set/natalie_portman_test2.jpg'
    # # test_fname = '/home/tomas/Dropbox/Data/celeb_dataset/celebrities/natalie_portman/natalie_portman_05.jpg'
    test_im = cv2.imread(test_fname)

    face_size = (30, 30)
    n_epochs = 200

    seed = 7
    np.random.seed(seed)

    # (X_train, y_train), (X_test, y_test) = mnist.load_data()

    # # extracting faces and writing to files
    # data2files(data_dir, face_cascade, res_file_path, label_dict_path)

    #
    # data_file = gzip.open(res_file_path, 'rb', compresslevel=1)
    # dataset = pickle.load(data_file)
    # data_file.close()
    # label_file = gzip.open(label_dict_path, 'rb', compresslevel=1)
    # label_dict = pickle.load(label_file)
    # label_file.close()
    #
    # data2TF_records(dataset, label_dict, tfrecords_path)

    # TRAIN AND PREDICT -----------------------
    run(data_dir, res_file_path, face_cascade, model_topol_path, model_weights_path, test_im, n_epochs=n_epochs, face_size=face_size)

    # LOAD AND PREDICT ------------------------
    model = load_model(model_topol_path, model_weights_path)
    query = detect_face(test_im, face_cascade)
    query = cv2.resize(query, dsize=face_size)
    query = query.reshape(1, 1, face_size[0], face_size[1]).astype('float32')
    preds = model.predict(query)[0]
    print preds

    data_file = gzip.open(res_file_path, 'rb', compresslevel=1)
    dataset = pickle.load(data_file)
    data_file.close()
    Y = [x['label'] for x in dataset]
    # encode class values as integers
    encoder = LabelEncoder()
    encoder.fit(Y)
    # encoded_Y = encoder.transform(Y)
    results = []
    k = 5
    sorted_idx = np.argsort(preds)[::-1][:5]
    print sorted_idx
    for i, idx in enumerate(sorted_idx):
        label = encoder.inverse_transform(idx)
        results.append((label, preds[idx]))
        print '#%i: %s, %i%%' % (i, label, int(100 * preds[idx]))

    plt.show()