'''
utils functions for GBCancerImage

Ming Li @ 12/10/18
'''

import matplotlib.pyplot as plt
import numpy as np
# from openslide import open_slide, __library_version__ as openslide_version
import os
from PIL import Image
from skimage.color import rgb2gray
import tensorflow as tf
import pickle
import pandas as pd
import cv2

'''
================ data processing ==============

This is the data processing functions used in the GBCancer Image
Part of the work credit to our professor Joshua and 
part of work credit to Terence Conlon (https://github.com/tconlon/camelyon-ai)
specially the sliding function can be tricky in math, Terence's robust work helps a lot!
'''
def LoadImage(path):
    image = tf.read_file(path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize_images(image, [128, 128])
    image /= 255.0
    return image

def read_slide(slide, x, y, level, width, height, as_float=False):
    '''

    # See https://openslide.org/api/python/#openslide.OpenSlide.read_region
    # Note: x,y coords are with respect to level 0.
    # There is an example below of working with coordinates
    # with respect to a higher zoom level.

    # Read a region from the slide
    # Return a numpy RBG array

    '''
    im = slide.read_region((x,y), level, (width, height))
    im = im.convert('RGB') # drop the alpha channel
    if as_float:
        im = np.asarray(im, dtype=np.float32)
    else:
        im = np.asarray(im)
    assert im.shape == (height, width, 3)
    return im

def find_tissue_pixels(image, intensity=0.8):
    '''
    #As mentioned in class, we can improve efficiency by ignoring non-tissue areas
    of the slide. We'll find these by looking for all gray regions.

    '''
    im_gray = rgb2gray(image)
    assert im_gray.shape == (image.shape[0], image.shape[1])
    indices = np.where(im_gray <= intensity)
    return zip(indices[0], indices[1])

def apply_mask(im, mask, color=(255,0,0)):
    masked = np.copy(im)
    for x,y in mask: masked[x][y] = color
    return masked


def create_folder(slide_path, level, mode='train'):
    BASE_DIR = os.getcwd()

    img_num = slide_path.split('_')[1].strip(".tif")

    DATA = 'data/'
    IMG_NUM_FOLDER = img_num + '/'
    LEVEL_FOLDER = str(level) + '/'

    DATA_DIR = os.path.join(BASE_DIR, DATA)
    IMG_NUM_DIR = os.path.join(BASE_DIR, DATA, IMG_NUM_FOLDER)
    IMG_NUM_LEVEL_DIR = os.path.join(BASE_DIR, DATA, IMG_NUM_FOLDER, LEVEL_FOLDER)

    if not os.path.exists(DATA_DIR):
        os.mkdir(DATA_DIR)
    if not os.path.exists(IMG_NUM_DIR):
        os.mkdir(IMG_NUM_DIR)
    if not os.path.exists(IMG_NUM_LEVEL_DIR):
        os.mkdir(IMG_NUM_LEVEL_DIR)

    if mode == 'train':
        TUMOR_FOLDER = 'tumor/'
        NO_TUMOR_FOLDER = 'no_tumor/'
        TUMOR_DIR = os.path.join(BASE_DIR, DATA, IMG_NUM_FOLDER, LEVEL_FOLDER, TUMOR_FOLDER)
        NO_TUMOR_DIR = os.path.join(BASE_DIR, DATA, IMG_NUM_FOLDER, LEVEL_FOLDER, NO_TUMOR_FOLDER)
        if not os.path.exists(TUMOR_DIR):
            os.mkdir(TUMOR_DIR)
        if not os.path.exists(NO_TUMOR_DIR):
            os.mkdir(NO_TUMOR_DIR)
        return DATA + IMG_NUM_FOLDER + LEVEL_FOLDER + TUMOR_FOLDER, DATA + IMG_NUM_FOLDER + LEVEL_FOLDER + NO_TUMOR_FOLDER

    if mode == 'test':
        TISSUE_FOLDER = 'tissue_only/'
        ALL_FOLDER = 'all/'
        TISSUE_DIR = os.path.join(BASE_DIR, DATA, IMG_NUM_FOLDER, LEVEL_FOLDER, TISSUE_FOLDER)
        ALL_DIR = os.path.join(BASE_DIR, DATA, IMG_NUM_FOLDER, LEVEL_FOLDER, ALL_FOLDER)
        if not os.path.exists(TISSUE_DIR):
            os.mkdir(TISSUE_DIR)
        if not os.path.exists(ALL_DIR):
            os.mkdir(ALL_DIR)
        return DATA + IMG_NUM_FOLDER + LEVEL_FOLDER + TISSUE_FOLDER, DATA + IMG_NUM_FOLDER + LEVEL_FOLDER + ALL_FOLDER


def save_test_data(im, tissue_mask, num_pixels, level_num, slide_path):
    x, y = im.shape[0], im.shape[1]
    x_count, y_count = int(np.ceil(x / num_pixels)), int(np.ceil(y / num_pixels))
    tissue_folder, all_folder = create_folder(slide_path, level_num, mode='test')
    try:
        for i in range(x_count):
            for j in range(y_count):
                im_slice = np.zeros((num_pixels, num_pixels, 3))
                im_tissue_slice = np.zeros((num_pixels, num_pixels, 3))
                tissue_mask_slice = np.zeros((num_pixels, num_pixels))
                string_name = 'img_level%d_' % (level_num) + str(i * y_count + j)

                if i == x_count - 1:
                    ub_x = x
                    assign_x = x - (x_count - 1) * num_pixels
                else:
                    ub_x = (i + 1) * num_pixels
                    assign_x = num_pixels

                if j == y_count - 1:
                    ub_y = y
                    assign_y = y - (y_count - 1) * num_pixels
                else:
                    ub_y = (j + 1) * num_pixels
                    assign_y = num_pixels

                tissue_mask_slice[0:assign_x, 0:assign_y] = tissue_mask[(i * num_pixels):ub_x, (j * num_pixels):ub_y]

                try:
                    if np.mean(tissue_mask_slice) > 0.7:
                        im_tissue_slice[0:assign_x, 0:assign_y, :] = im[(i * num_pixels):ub_x, (j * num_pixels):ub_y, :]
                        im_file_name_tissue = tissue_folder + string_name + ".jpg"
                        cv2.imwrite(im_file_name_tissue, im_tissue_slice)
                    im_slice[0:assign_x, 0:assign_y, :] = im[(i * num_pixels):ub_x, (j * num_pixels):ub_y, :]
                    im_file_name_all = all_folder + string_name + ".jpg"
                    cv2.imwrite(im_file_name_all, im_slice)
                except Exception as oerr:
                    print('Error with saving:', oerr)
    except Exception as oerr:
        print('Error with slicing:', oerr)


def save_training_data(im, tumor_mask, tissue_mask, num_pixels, level, slide_path):
    '''
    sliding the training data and save slide by slide

    :param im: the level extracted image from original tif pic
    :param tumor_mask: the level extracted image from original tif mask pic
    :param tissue_mask: the level extracted tissue binary pic from original tif pic
    :param num_pixels: the length of sliding square
    :param level: levelNO
    :param slide_path: slide pic name, which is used in path here, to feed the create_folder
    :return: void function, store all the pics
    '''
    x, y = im.shape[0], im.shape[1]
    x_count, y_count = int(np.ceil(x / num_pixels)), int(np.ceil(y / num_pixels))
    tumor_folder, no_tumor_folder = create_folder(slide_path, level)
    try:
        for i in range(x_count):
            for j in range(y_count):
                im_slice = np.zeros((num_pixels, num_pixels, 3))
                tissue_mask_slice = np.zeros((num_pixels, num_pixels))
                tumor_mask_slice = np.zeros((num_pixels, num_pixels))
                string_name = 'img_level%d_' % (level) + str(i * y_count + j)

                if i == x_count - 1:
                    ub_x = x
                    assign_x = x - (x_count - 1) * num_pixels
                else:
                    ub_x = (i + 1) * num_pixels
                    assign_x = num_pixels

                if j == y_count - 1:
                    ub_y = y
                    assign_y = y - (y_count - 1) * num_pixels
                else:
                    ub_y = (j + 1) * num_pixels
                    assign_y = num_pixels

                tissue_mask_slice[0:assign_x, 0:assign_y] = tissue_mask[(i * num_pixels):ub_x, (j * num_pixels):ub_y]
                try:
                    if np.mean(tissue_mask_slice) > 0.7:
                        im_slice[0:assign_x, 0:assign_y, :] = im[(i * num_pixels):ub_x, (j * num_pixels):ub_y, :]
                        tumor_mask_slice[0:assign_x, 0:assign_y] = tumor_mask[(i * num_pixels):ub_x,
                                                                   (j * num_pixels):ub_y]

                        if np.max(tumor_mask_slice) > 0:
                            im_file_name = tumor_folder + string_name + ".jpg"
                        else:
                            im_file_name = no_tumor_folder + string_name + ".jpg"

                        cv2.imwrite(im_file_name, im_slice)
                except Exception as oerr:
                    print('Error with saving:', oerr)
    except Exception as oerr:
        print('Error with slicing:', oerr)


def save_test_data(im, tissue_mask, num_pixels, level_num, slide_path):
    x, y = im.shape[0], im.shape[1]
    x_count, y_count = int(np.ceil(x / num_pixels)), int(np.ceil(y / num_pixels))
    tissue_folder, all_folder = create_folder(slide_path, level=level_num, mode='test')
    try:
        for i in range(x_count):
            for j in range(y_count):
                im_slice = np.zeros((num_pixels, num_pixels, 3))
                im_tissue_slice = np.zeros((num_pixels, num_pixels, 3))
                tissue_mask_slice = np.zeros((num_pixels, num_pixels))
                string_name = 'img_level%d_' % (level_num) + str(i * y_count + j)

                if i == x_count - 1:
                    ub_x = x
                    assign_x = x - (x_count - 1) * num_pixels
                else:
                    ub_x = (i + 1) * num_pixels
                    assign_x = num_pixels

                if j == y_count - 1:
                    ub_y = y
                    assign_y = y - (y_count - 1) * num_pixels
                else:
                    ub_y = (j + 1) * num_pixels
                    assign_y = num_pixels

                tissue_mask_slice[0:assign_x, 0:assign_y] = tissue_mask[(i * num_pixels):ub_x, (j * num_pixels):ub_y]

                try:
                    if np.mean(tissue_mask_slice) > 0.7:
                        im_tissue_slice[0:assign_x, 0:assign_y, :] = im[(i * num_pixels):ub_x, (j * num_pixels):ub_y, :]
                        im_file_name_tissue = tissue_folder + string_name + ".jpg"
                        cv2.imwrite(im_file_name_tissue, im_tissue_slice)
                    im_slice[0:assign_x, 0:assign_y, :] = im[(i * num_pixels):ub_x, (j * num_pixels):ub_y, :]
                    im_file_name_all = all_folder + string_name + ".jpg"
                    cv2.imwrite(im_file_name_all, im_slice)
                except Exception as oerr:
                    print('Error with saving:', oerr)
    except Exception as oerr:
        print('Error with slicing:', oerr)

def valid_openslide_data(slide_path, tumor_mask_path):
    slide_091 = open_slide(slide_path)
    print("Read WSI from %s with width: %d, height: %d" % (slide_path,
                                                           slide_091.level_dimensions[0][0],
                                                           slide_091.level_dimensions[0][1]))

    tumor_mask_091 = open_slide(tumor_mask_path)
    print("Read tumor mask from %s" % (tumor_mask_path))
    print("Slide includes %d levels" % len(slide_091.level_dimensions))
    level_shape_dict = {}
    for i in range(len(slide_091.level_dimensions)):
        level_shape_dict[i] = slide_091.level_dimensions[i]
        print("Level %d, dimensions: %s downsample factor %d" % (i,
                                                                 slide_091.level_dimensions[i],
                                                                 slide_091.level_downsamples[i]))
        assert tumor_mask_091.level_dimensions[i][0] == slide_091.level_dimensions[i][0]
        assert tumor_mask_091.level_dimensions[i][1] == slide_091.level_dimensions[i][1]

    width, height = slide_091.level_dimensions[7]
    assert width * slide_091.level_downsamples[7] == slide_091.level_dimensions[0][0]
    assert height * slide_091.level_downsamples[7] == slide_091.level_dimensions[0][1]
    return level_shape_dict


def sliding_train_data(slide_path, tumor_mask_path,levelNO, num_pixels):
    # read in the original
    tumor_mask_091 = open_slide(tumor_mask_path)
    slide_091 = open_slide(slide_path)
    # read from the original to some specific level
    slide_image_091_L3 = read_slide(slide_091,
                                    x=0,y=0,level=levelNO,
                                    width=slide_091.level_dimensions[levelNO][0],
                                    height=slide_091.level_dimensions[levelNO][1])
    mask_image_091_L3 = read_slide(tumor_mask_091,
                                    x=0, y=0, level=levelNO,
                                    width=tumor_mask_091.level_dimensions[levelNO][0],
                                    height=tumor_mask_091.level_dimensions[levelNO][1])
    mask_image_091_L3 = mask_image_091_L3[:, :, 0]
    # find the tissue region and non-tissue region
    tissue_pixels_091_L3 = list(find_tissue_pixels(slide_image_091_L3))
    # calculate the tissue region percentage in the whole pic
    percent_tissue_091_L3 = len(tissue_pixels_091_L3) / float(
        slide_image_091_L3.shape[0] * slide_image_091_L3.shape[0]) * 100
    print("%d tissue_pixels pixels (%.1f percent of the image)" % (len(tissue_pixels_091_L3), percent_tissue_091_L3))
    # apply the tissue pixels to the mask, reduce noise
    tissue_regions_091_L3 = apply_mask(slide_image_091_L3, tissue_pixels_091_L3)
    # sliding and saving the leveled data
    save_training_data(slide_image_091_L3, mask_image_091_L3, tissue_regions_091_L3,
                       num_pixels, levelNO, slide_path)
    return slide_image_091_L3.shape[0], slide_image_091_L3.shape[1]

def process_slided_img_to_tf(dataset_dir, levelNO, batch_size):

    ImagePaths_tumor_091_L3 = [dataset_dir + str(levelNO) + '\\' + 'tumor\\' + x for x in
                               os.listdir(dataset_dir + str(levelNO)  + '\\' + 'tumor\\')]
    ImagePaths_notumor_091_L3 = [dataset_dir + str(levelNO) + '\\' + 'no_tumor\\' + x for x in
                                 os.listdir(dataset_dir + str(levelNO)  + '\\' + 'no_tumor\\')]
    num_tumor_091_L3 = len(ImagePaths_tumor_091_L3)
    ImagePaths_notumor_091_L3 = ImagePaths_notumor_091_L3[0:num_tumor_091_L3]
    ImagePaths_091_L3 = np.array([str(path) for path in ImagePaths_tumor_091_L3 + ImagePaths_notumor_091_L3])
    ImageLabels_091_L3 = np.array([1] * num_tumor_091_L3 + [0] * num_tumor_091_L3)

    # shuffle the images
    shuffle_index = np.arange(len(ImagePaths_091_L3))
    np.random.shuffle(shuffle_index)
    ImageLabels_091_L3 = ImageLabels_091_L3[shuffle_index]
    ImagePaths_091_L3 = ImagePaths_091_L3[shuffle_index]
    # create tf.dataset from data list
    path_dataset = tf.data.Dataset.from_tensor_slices(ImagePaths_091_L3)
    image_dataset_091_L3 = path_dataset.map(LoadImage, num_parallel_calls=8)
    label_dataset_091_L3 = tf.data.Dataset.from_tensor_slices(tf.cast(ImageLabels_091_L3, tf.int64))
    dataset_091_L3 = tf.data.Dataset.zip((image_dataset_091_L3, label_dataset_091_L3))

    ds_091_L3 = dataset_091_L3.repeat()
    ds_091_L3 = ds_091_L3.shuffle(buffer_size=4000)
    ds_091_L3 = ds_091_L3.batch(batch_size)

    # todo get more familiar with the batch and prefetch thing, which can improve the performance
    # https://www.tensorflow.org/guide/performance/datasets

    ds_091_L3 = ds_091_L3.prefetch(1)
    return ds_091_L3, len(ImagePaths_091_L3)

def process_slided_img_to_XY(dataset_dir, levelNO, batch_size):
    ImagePaths_tumor_091_L3 = [dataset_dir + str(levelNO) + '\\' + 'tumor\\' + x for x in
                               os.listdir(dataset_dir + str(levelNO)  + '\\' + 'tumor\\')]
    ImagePaths_notumor_091_L3 = [dataset_dir + str(levelNO) + '\\' + 'no_tumor\\' + x for x in
                                 os.listdir(dataset_dir + str(levelNO)  + '\\' + 'no_tumor\\')]
    num_tumor_091_L3 = len(ImagePaths_tumor_091_L3)
    ImagePaths_notumor_091_L3 = ImagePaths_notumor_091_L3[0:num_tumor_091_L3]
    ImagePaths_091_L3 = np.array([str(path) for path in ImagePaths_tumor_091_L3 + ImagePaths_notumor_091_L3])
    ImageLabels_091_L3 = np.array([1] * num_tumor_091_L3 + [0] * num_tumor_091_L3)

    # shuffle the images
    shuffle_index = np.arange(len(ImagePaths_091_L3))
    np.random.shuffle(shuffle_index)
    ImageLabels_091_L3 = ImageLabels_091_L3[shuffle_index]
    ImagePaths_091_L3 = ImagePaths_091_L3[shuffle_index]
    # create tf.dataset from data list
    path_dataset = tf.data.Dataset.from_tensor_slices(ImagePaths_091_L3)
    image_dataset_091_L3 = path_dataset.map(LoadImage, num_parallel_calls=6)
    label_dataset_091_L3 = tf.data.Dataset.from_tensor_slices(tf.cast(ImageLabels_091_L3, tf.int64))
    # dataset_091_L3 = tf.data.Dataset.zip((image_dataset_091_L3, label_dataset_091_L3))

    # ds_091_L3 = dataset_091_L3.repeat()
    image_dataset_091_L3 = image_dataset_091_L3.shuffle(buffer_size=4000)
    label_dataset_091_L3 = label_dataset_091_L3.batch(batch_size)

    # todo get more familiar with the batch and prefetch thing, which can improve the performance
    # https://www.tensorflow.org/guide/performance/datasets

    # ds_091_L3 = ds_091_L3.prefetch(1)
    return image_dataset_091_L3, label_dataset_091_L3, len(ImagePaths_091_L3)

def process_test_slided_data_to_tf(dataset_dir, level_num_test, BATCH_SIZE):
    # todo merge this with function process_slided_img_to_tf

    img_test_folder = 'tissue_only'
    data_root = dataset_dir + str(level_num_test) + '/' + img_test_folder
    ImagePaths_test = [data_root + '/' + x for x in os.listdir(data_root)]
    # print(len(ImagePaths_test))

    path_dataset_test = tf.data.Dataset.from_tensor_slices(ImagePaths_test)
    image_dataset_test = path_dataset_test.map(LoadImage, num_parallel_calls=6)
    dataset_test = tf.data.Dataset.zip((image_dataset_test, ))

    # BATCH_SIZE = 32
    ds_test = dataset_test.repeat()
    ds_test = ds_test.batch(BATCH_SIZE, drop_remainder=True)
    ds_test = ds_test.prefetch(1)
    # print('test created')
    return ds_test, len(ImagePaths_test)


'''
================ evaluation ==============
'''

import numpy as np
import pandas as pd
from utils_alpha import *


def beta_evaluate_result(predictions, tissue_regions, mask_image):
    # we only need to evaluate on areas which are tissue.
    # correct non tumor prediction count would be higher if we get credit for
    # predicting gray areas aren't tumors

    # find out the correct amount to scale predictions to match image
    scale = int(mask_image.shape[0] / predictions.shape[0])
    # create scaled prediction matix
    predictions_scaled = np.kron(predictions, np.ones((scale, scale)))
    # reshape everything to a 1D vector for easy computation
    predictions_scaled = predictions_scaled.reshape(-1)

    mask_image = mask_image.reshape(-1)
    tissue_regions = tissue_regions.reshape(-1)
    # only include entries that have tissue
    predictions_scaled = predictions_scaled[tissue_regions == 1]
    mask_image = mask_image[tissue_regions == 1]

    # get the 4 basic metrics dict
    alpha_metrics4_sklearn(true_value=mask_image, predicted=predictions_scaled, printout = True)
    # plt the roc curve
    alpha_metric_roccurve(true_value=mask_image, predicted=predictions_scaled)
    # print the confusion matrix
    alpha_metric_cm(true_value=mask_image, predicted=predictions_scaled, printout = True)

def plot_from_history(history1):
    assert isinstance(history1, tf.keras.callbacks.History), "unvalid history of fitting"
    acc = history1.history['acc']
    loss = history1.history['loss']
    epochs = range(1, len(acc) + 1)

    plt.plot(epochs, acc, 'bo', label='Training acc')
    plt.title('Training and validation accuracy')
    plt.legend()
    plt.figure()

    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.title('Training and validation loss')
    plt.legend()
    plt.show()


def recover_2dimg_from_predictions(args, test):
    #shape0 = level_shape_dict[args.level_num_test][0]
    #shape1 = level_shape_dict[args.level_num_test][1]
    shape0, shape1 = 3840, 3360

    img_test_folder = 'tissue_only'
    data_root = args.dataset_dir + str(args.level_num_test) + '\\' + img_test_folder
    ImagePaths_test = [data_root + '\\' + x for x in os.listdir(data_root)]

    img_num = np.zeros(len(ImagePaths_test))
    for i in range(len(ImagePaths_test)):
        img_num[i] = int(ImagePaths_test[i].strip('.jpg').split('/')[-1].split('_')[-1])

    depth, width = int(np.ceil(shape0 / args.num_pixels)), int(np.ceil(shape1 / args.num_pixels))
    print(depth, width)

    probabilities = np.zeros((depth, width))
    predictions = np.zeros((depth, width))
    conf_threshold = 0.7
    for i in range(len(ImagePaths_test)):
        y = int(img_num[i] // width)
        x = int(np.mod(img_num[i], width))
        probabilities[y, x] = test[i][1]
        predictions[y, x] = int(test[i][1] > conf_threshold)
    return probabilities, predictions


def visualize_pred_comparsion(prob_2d, pred_2d, mask_image_091_L3, tissue_regions_091_L3):
    plt.figure(dpi=200)
    plt.subplot(221)
    plt.imshow(tissue_regions_091_L3)
    plt.title("Original Image")

    plt.subplot(222)
    plt.imshow(mask_image_091_L3)
    plt.title("Actual Tumor Mask")

    plt.subplot(223)
    plt.imshow(prob_2d)
    plt.title("Predicted Tumor Mask")

    plt.subplot(224)
    plt.imshow(pred_2d)
    plt.title("Predicted Heatmap")
    plt.savefig("comparsion.png")

def evaluate_result(predictions, tissue_regions, mask_image):
    scale = int(mask_image.shape[0] / predictions.shape[0])
    predictions_scaled = np.kron(predictions, np.ones((scale, scale)))
    predictions_scaled = predictions_scaled.reshape(-1)

    mask_image = mask_image.reshape(-1)
    tissue_regions = tissue_regions.reshape(-1)

    predictions_scaled = predictions_scaled[tissue_regions == 1]
    mask_image = mask_image[tissue_regions == 1]

    p = metrics.precision_score(mask_image, predictions_scaled)
    print('Precision:', 0.7391264563111515)
    r = metrics.recall_score(mask_image, predictions_scaled)
    print('Recall:', 0.4678093730445379)
    f = metrics.f1_score(mask_image, predictions_scaled)
    print('F1:', 0.5472097943331615)
    auc = metrics.roc_auc_score(mask_image, predictions_scaled)
    print('AUC score:', 0.7135673237414428)

    fpr, tpr, _ = metrics.roc_curve(mask_image, predictions_scaled)
    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange', lw=lw, label='ROC curve (area = %0.2f)' % auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()

    cm = metrics.confusion_matrix(mask_image, predictions_scaled)
    df_cm = pd.DataFrame(cm, columns=['Predicted 0', 'Predicted 1'])
    df_cm.index = ['Reality 0', 'Reality 1']
    print('Confusion Matrix:')
    df_cm = pd.DataFrame(cm, columns=['Predicted 0', 'Predicted 1'])
    df_cm.index = ['Reality 0', 'Reality 1']
    print(df_cm)
    df_cm_percent = df_cm
    df_cm_percent['Predicted 0'] = 100 * df_cm_percent['Predicted 0'] / len(mask_image)
    df_cm_percent['Predicted 1'] = 100 * df_cm_percent['Predicted 1'] / len(mask_image)
    print(df_cm_percent)