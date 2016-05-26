import numpy as np
from skimage import io, color, exposure, transform
from sklearn.cross_validation import train_test_split
import os
import glob
import h5py

from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, model_from_json
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.optimizers import SGD
from keras.utils import np_utils

NUM_CLASSES = 43
IMG_SIZE = 48

def preprocess_img(img):
    # Histogram normalization in y
    hsv = color.rgb2hsv(img)
    hsv[:,:,2] = exposure.equalize_hist(hsv[:,:,2])
    img = color.hsv2rgb(hsv)

    # central scrop
    min_side = min(img.shape[:-1])
    centre = img.shape[0]//2, img.shape[1]//2
    img = img[centre[0]-min_side//2:centre[0]+min_side//2,
              centre[1]-min_side//2:centre[1]+min_side//2,
              :]

    # rescale to standard size
    img = transform.resize(img, (IMG_SIZE, IMG_SIZE))

    # roll color axis to axis 0
    img = np.rollaxis(img,-1)

    return img


def get_class(img_path):
    return int(img_path.split('/')[-2])


def preprocess_all_images(root_dir, is_labeled=False):
    
    imgs = []
    if is_labeled:
        all_img_paths = glob.glob(os.path.join(root_dir, '*/*.ppm'))
        labels = []
    else:
        all_img_paths = glob.glob(os.path.join(root_dir, '*.ppm'))

    for img_path in all_img_paths:
        try:
            img = preprocess_img(io.imread(img_path))
            imgs.append(img)
            if is_labeled: 
                label = get_class(img_path)
                labels.append(label)
            
            if len(imgs)%100 == 0: print("{}/{}".format(len(imgs), len(all_img_paths)))
        except (IOError, OSError):
            print('missed', img_path)
            pass

    imgs = np.array(imgs, dtype='float32')
    if is_labeled:
        labels = np.eye(NUM_CLASSES, dtype='uint8')[labels]
        return imgs, labels
    else:
        return imgs


def cnn_model():
    model = Sequential()

    model.add(Convolution2D(32, 3, 3, border_mode='same', input_shape=(3, IMG_SIZE, IMG_SIZE)))
    model.add(Activation('relu'))
    model.add(Convolution2D(32, 3, 3))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Convolution2D(64, 3, 3, border_mode='same'))
    model.add(Activation('relu'))
    model.add(Convolution2D(64, 3, 3))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Convolution2D(128, 3, 3, border_mode='same'))
    model.add(Activation('relu'))
    model.add(Convolution2D(128, 3, 3))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(NUM_CLASSES))
    model.add(Activation('softmax'))

    return model


def train(X, Y, batch_size=32, nb_epoch=30, data_augmentation=False):
    model = cnn_model()
    batch_size = 32
    nb_epoch = 30
    data_augmentation = False

    # let's train the model using SGD + momentum (how original).
    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy',
              optimizer=sgd,
              metrics=['accuracy'])


    if not data_augmentation:
        print('Not using data augmentation.')
        model.fit(X, Y,
                  batch_size=batch_size,
                  nb_epoch=nb_epoch,
                  validation_split=0.33,
                  shuffle=True)
    else:
        print('Using real-time data augmentation.')
        X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.33, random_state=42)

        datagen = ImageDataGenerator(featurewise_center=False, 
                                    featurewise_std_normalization=False, 
                                    width_shift_range=0.1,
                                    height_shift_range=0.1,
                                    zoom_range=0.2,
                                    shear_range=0.1,
                                    rotation_range=10.,)
        datagen.fit(X_train)

        model.fit_generator(datagen.flow(X_train, Y_train, batch_size=batch_size),
                            samples_per_epoch=X_train.shape[0],
                            nb_epoch=nb_epoch,
                            validation_data=(X_val, Y_val))


    json_string = model.to_json()
    open('my_model_architecture.json', 'w').write(json_string)
    model.save_weights('my_model_weights.h5')

    return model


if __name__ == '__main__':
    try:
        X, Y = h5py.File('X.h5')['imgs'][:], h5py.File('X.h5')['labels'][:]
    except (OSError, IOError):
        with h5py.File('X.h5','w') as hf:
            imgs, labels = preprocess_all_images('GTSRB/Final_Training/Images/', is_labeled=True)
            hf.create_dataset('imgs', data=imgs)
            hf.create_dataset('labels', data=labels)
        
        with h5py.File('X_test.h5','w') as hf:
            imgs, labels = preprocess_all_images(os.path.join('GTSRB/Final_Testing/Images/'), is_labeled=False)
            hf.create_dataset('imgs', data=imgs)
            hf.create_dataset('labels', data=labels)


    try:
        model = model_from_json(open('my_model_architecture.json').read())
        model.load_weights('my_model_weights.h5')
    except:
        model = train(X, Y, nb_epoch=15)
