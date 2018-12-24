import logging
# todo add logging module to this program
import argparse
import os
import time
from utils import *
from module import *
# use this line to make sure it works with jupyter notebook or colab
import sys; sys.argv=['']; del sys

parser = argparse.ArgumentParser(description='')
parser.add_argument('--slide_path_091', dest='slide_path_091', default='tumor_091.tif', help='The img used for modelling')
parser.add_argument('--tumor_mask_path_091', dest='tumor_mask_path_091', default='tumor_091_mask.tif', help='The img used for modelling')
parser.add_argument('--dataset_dir', dest='dataset_dir', default='data\\091\\', help='path of the dataset')
parser.add_argument('--levelNO', dest='levelNO', type = int, default=3, help='level from 1 to 8')
parser.add_argument('--num_pixels', dest='num_pixels', type = int,  default=64, help='the length of sliding square')
parser.add_argument('--batch_size', dest='BATCH_SIZE', type = int,  default=32, help='# images in batch')
parser.add_argument('--epoch', dest='epoch', type=int, default=8, help='# of epoch')
parser.add_argument('--base_model', dest='base_model_name', default='VGG16', help='name of base model')
parser.add_argument('--build_shape', dest='build_shape', default=tf.TensorShape([None, 128, 128, 3]), help='build shape for VGG16 based model')
parser.add_argument('--period', dest='period', type=int, default=5, help='save a model every period epochs')
parser.add_argument('--level_num_test', dest='level_num_test', type=int, default=2, help='level from 1 to 8')

parser.add_argument('--lr', dest='lr', type=float, default=0.0002, help='initial learning rate for adam')

parser.add_argument('--phase', dest='phase', default='test', help='train, test')
parser.add_argument('--continue_train', dest='continue_train', type=bool, default=False, help='if continue training, load the latest model: 1: true, 0: false')
parser.add_argument('--checkpoint_dir', dest='checkpoint_dir', default='./checkpoint', help='models are saved here')
parser.add_argument('--sample_dir', dest='sample_dir', default='./sample', help='sample are saved here')
parser.add_argument('--test_dir', dest='test_dir', default='./test', help='test sample are saved here')
parser.add_argument('--L1_lambda', dest='L1_lambda', type=float, default=10.0, help='weight on L1 term in objective')
args = parser.parse_args()

def main():

    # check and create test and checkpoint folders
    if not os.path.exists(args.checkpoint_dir):
        os.makedirs(args.checkpoint_dir)
    if not os.path.exists(args.test_dir):
        os.makedirs(args.test_dir)


    # check the dataset exist
    if not os.path.exists(args.slide_path_091):
        print('please run the download_dataset.sh to download the dataset')
    else:
        pass



    if args.phase == 'train':

        # check the slided dataset dir exist
        if not os.path.exists(args.dataset_dir):
            print('Begin training data segmentation:')
            start_time = time.time()
            level_shape_dict = valid_openslide_data(slide_path=args.slide_path_091,
                                 tumor_mask_path=args.tumor_mask_path_091)
            _, _ = sliding_train_data(slide_path=args.slide_path_091, tumor_mask_path=args.tumor_mask_path_091, levelNO=args.levelNO,num_pixels=args.num_pixels)
            print('time taken for sliding the training data:', time.time() - start_time)
        else:
            print('data folder already exists')

        # get the training data from slided directory
        ds_091_L3, img_count = process_slided_img_to_tf(dataset_dir=args.dataset_dir, levelNO=args.levelNO, batch_size=args.BATCH_SIZE)
        # print('process slided imgs to tf input data. done!')

        model1 = create_model()
        history, model = train_model(args, model1, ds_091_L3, img_count)
        plot_from_history(history)

    else:
        print('reload the latest model from checkpoint')
        model_test = restore_model_from_latest_ckpt(args)

        if not os.path.exists(args.dataset_dir + str(args.level_num_test)):
            shape0, shape1 = sliding_train_data(slide_path=args.slide_path_091, tumor_mask_path=args.tumor_mask_path_091,\
                                                levelNO=args.levelNO, num_pixels=args.num_pixels)
        ds_test, img_count_test = process_test_slided_data_to_tf(dataset_dir = args.dataset_dir, \
                                                                 level_num_test = args.level_num_test, BATCH_SIZE = 32)
        try:
            predictions = model_test.predict(ds_test, steps=int(np.ceil(img_count_test / 32)))
        except Exception as e:
            #print('model predict failed')
            print('predict done')

        # prob_2d, pred_2d = recover_2dimg_from_predictions(args, predictions)
        pred_2d, prob_2d, mask_image_091_L3, tissue_regions_091_L3 = load_pickles()

        visualize_pred_comparsion(prob_2d, pred_2d, mask_image_091_L3, tissue_regions_091_L3)
        # beta_evaluate_result(pred_2d, tissue_regions_091_L3, mask_image_091_L3)
        evaluate_result(pred_2d, tissue_regions_091_L3, mask_image_091_L3)


if __name__ == "__main__":
    main()