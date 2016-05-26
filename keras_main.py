import numpy as np
from skimage import io, color, exposure, transform
import os
import glob
import h5py

NUM_CLASSES = 43
IMGs_SIZE = 48

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

def preprocess_all_images(root_dir):
    all_img_paths = glob.glob(os.path.join(root_dir, '*/*.ppm'))
    imgs = []
    labels = []

    for img_path in all_img_paths:
        try:
            img = preprocess_img(io.imread(img_path))
            label = get_class(img_path)
            imgs.append(img)
            labels.append(label)
            if len(imgs)%100 == 0: print("{}/{}".format(len(imgs), len(all_img_paths)))
        except (IOError, OSError):
            print('missed', img_path)
            pass

    imgs = np.array(imgs, dtype='float32')
    labels = np.eye(NUM_CLASSES)[labels]

    return imgs, labels


with h5py.File('X_train.h5','w') as hf:
    imgs, labels = preprocess_all_images('GTSRB/Final_Training/Images/train')
    hf.create_dataset('imgs', data=imgs)
    hf.create_dataset('labels', data=labels)

with h5py.File('X_val.h5','w') as hf:
    imgs, labels = preprocess_all_images('GTSRB/Final_Training/Images/val')
    hf.create_dataset('imgs', data=imgs)
    hf.create_dataset('labels', data=labels)


