import numpy as np
import h5py
from random import shuffle
import glob
import cv2
import os



def create_hd5file(data_path, hdf5_path, image_shape, shuffle_data = True):
    #shuffle_data = True  # shuffle the addresses before saving
    #hdf5_path = '/home/kadatta/Downloads/liveness_detection/code/train_dataset_casia.hdf5'  # address to where you want to save the hdf5 file
    #data_path = '/home/kadatta/Downloads/liveness_detection/casia_dataset/Casia_new_traintest/Faces_train_data_42subjects_ipsoft_casia_dataug/%s/*.jpg'

    data_path = os.path.join(data_path, '%s/*')
    # read addresses and labels from the data_path folder
    train_addrs = []
    train_labels = []
    test_addrs = []
    test_labels = []

    for i,categ in enumerate(['train']):
        data = glob.glob(data_path%categ)
        print '%s = %s'%(categ, len(data))
        train_addrs += data
        train_labels += [i]*len(data)

    # to shuffle train data
    if shuffle_data:
        c = list(zip(train_addrs, train_labels))
        shuffle(c)
        train_addrs, train_labels = zip(*c)



    for i,categ in enumerate(['test']):
        data = glob.glob(data_path%categ)
        print '%s = %s'%(categ, len(data))
        test_addrs += data
        test_labels += [i]*len(data)
        print(data)

    # to shuffle test data
    if shuffle_data:
        c = list(zip(test_addrs, test_labels))
        shuffle(c)
        test_addrs, test_labels = zip(*c)



    width,height,channels = image_shape
    train_shape = (len(train_addrs), width, height, channels)
    test_shape = (len(test_addrs), width, height, channels)

    # open a hdf5 file and create earrays
    hdf5_file = h5py.File(hdf5_path, mode='w')

# train arrays...

    hdf5_file.create_dataset("train_X", train_shape, np.int8)
    hdf5_file.create_dataset("train_mean", train_shape[1:], np.float32)
    hdf5_file.create_dataset("train_Y", (len(train_addrs),), np.int8)
    hdf5_file["train_Y"][...] = train_labels

# test arrays...
    hdf5_file.create_dataset("test_X", test_shape, np.int8)
    hdf5_file.create_dataset("test_mean", test_shape[1:], np.float32)
    hdf5_file.create_dataset("test_Y", (len(test_addrs),), np.int8)
    hdf5_file["test_Y"][...] = test_labels


    # a numpy array to save the mean of the images
    mean_train = np.zeros(train_shape[1:], np.float32)
    mean_test = np.zeros(test_shape[1:], np.float32)


    # loop over train addresses
    for i in range(len(train_addrs)):
        # print how many images are saved every 1000 images
        if i % 1000 == 0 and i > 1:
            print 'Train data: {}/{}'.format(i, len(train_addrs))

        # read an image and resize to (224, 224)
        # cv2 load images as BGR, convert it to RGB
        addr = train_addrs[i]
        img = cv2.imread(addr)
        if img.shape != image_shape:
            img = cv2.resize(img, (width,height), interpolation=cv2.INTER_CUBIC)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


        # add any image pre-processing here


        # save the image and calculate the mean so far
        hdf5_file["train_X"][i, ...] = img[None]
        mean_train += img / float(len(train_labels))
        if 'cat.' in train_addrs[i]:
            # print('Cat {}' .format(i))
            hdf5_file["train_Y"][i] = 0
        elif 'dog.' in train_addrs[i]:
            # print('Dog {}' .format(i))
            hdf5_file["train_Y"][i] = 1
    # save the mean in the hdf5 file
    #hdf5_file["train_mean"][...] = mean_train





    # loop over test addresses
    for i in range(len(test_addrs)):
        # print how many images are saved every 500 images
        if i % 1000 == 0 and i > 1:
            print 'Test data: {}/{}'.format(i, len(test_addrs))

        # read an image and resize to (224, 224)
        # cv2 load images as BGR, convert it to RGB
        addr = test_addrs[i]
        img = cv2.imread(addr)
        if img.shape != image_shape:
            img = cv2.resize(img, (width,height), interpolation=cv2.INTER_CUBIC)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


        # add any image pre-processing here


        # save the image and calculate the mean so far
        hdf5_file["test_X"][i, ...] = img[None]
        mean_test += img / float(len(test_labels))
        if 'cat.' in test_addrs[i]:
            # print('Cat {}' .format(i))
            hdf5_file["test_Y"][i] = 0
        elif 'dog.' in test_addrs[i]:
            # print('Dog {}' .format(i))
            hdf5_file["test_Y"][i] = 1
    # save the mean in the hdf5 file
    #hdf5_file["test_mean"][...] = mean_test


    #close the hd5 file
    hdf5_file.close()



if __name__ == '__main__':
    #data_path = '/home/kadatta/Downloads/liveness_detection/casia_dataset/Casia_new_traintest/Faces_train_data_42subjects_ipsoft_casia_dataug/'
    data_path = '/home/sunising/Downloads/misc/cats_vs_dogs/Data/'

    # address to where you want to save the hdf5 file
    #hdf5_path = '/home/kadatta/Downloads/liveness_detection/code/hd5_files_dir/train_dataset_casiaipsoft_shuffled.h5'
    hdf5_path = '/home/sunising/Downloads/misc/cats_vs_dogs/Data/load_dataset.h5'

    image_shape = (160,160,3)
    shuffle_data=True #shuffle the data before saving
    create_hd5file(data_path, hdf5_path, image_shape, shuffle_data)


