import os

"""
PATHS
"""
OUTPUT_PATH = '.'
__TF_RECORD_ROOT_PATH = 'data'
TF_RECORD_TRAIN_PATH = os.path.join(__TF_RECORD_ROOT_PATH, 'train.tfrecord')
TF_RECORD_EVAL_PATH = os.path.join(__TF_RECORD_ROOT_PATH, 'eval.tfrecord')
TF_RECORD_TEST_PATH = os.path.join(__TF_RECORD_ROOT_PATH, 'test.tfrecord')

ROOT_PATH = '/mnt/data/dl/datasets/sdc/kitti/tracking/'
TRAIN_LABEL_PATH = '/mnt/data/dl/datasets/sdc/kitti/tracking/training/label_02'
TRAIN_IMAGE_PATH = '/mnt/data/dl/datasets/sdc/kitti/tracking/training/image_02'
TRAIN_LIDAR_PATH = '/mnt/data/dl/datasets/sdc/kitti/tracking/training/velodyne'
TEST_IMAGE_PATH = '/mnt/data/dl/datasets/sdc/kitti/tracking/testing/image_02'
TEST_LIDAR_PATH = '/mnt/data/dl/datasets/sdc/kitti/tracking/testing/velodyne'

"""
DATA
"""
LABEL_INDEX = {
    'Car': 1,
    'Van': 1,
    'Truck': 1,

    'Pedestrian': 2,
    'Person': 2,
    'Cyclist': 2,

    'Tram': -1,
    'Misc': -1,
    'DontCare': -1
}

"""
TRAINING
"""
EVAL_TRAIN_SPLIT = 0.8

if __name__ == '__main__':
    print("Nothing to do")
