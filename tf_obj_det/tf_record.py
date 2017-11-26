import os

import Constants
import get_data
import tensorflow as tf
from object_detection.utils import dataset_util


def generate_tf_record() -> None:
    """
    Converts the kitti dataset into Tensorflow record objects
    :return: None
    """
    # Remove the files if they already exist
    if os.path.exists(Constants.TF_RECORD_TRAIN_PATH):
        os.remove(Constants.TF_RECORD_TRAIN_PATH)
    if os.path.exists(Constants.TF_RECORD_EVAL_PATH):
        os.remove(Constants.TF_RECORD_EVAL_PATH)
    if os.path.exists(Constants.TF_RECORD_TEST_PATH):
        os.remove(Constants.TF_RECORD_TEST_PATH)

    # Generate both the records
    __generate_train_record()
    __generate_test_record()


def __generate_test_record():
    # Get the record writer ready
    test_writer = tf.python_io.TFRecordWriter('data/test.tfrecord')

    # Total number of sequences
    print("Loading the images and writing to testing record")
    num_sequences = len(get_data.get_test_sequences())
    for sequence in get_data.get_test_sequences():
        print("Starting test sequence:", sequence, "of", num_sequences)
        for frame in get_data.get_test_frames(sequence):
            img = get_data.load_test_image(sequence, frame)
            img_raw = img.tostring()
            filename = get_data.get_test_img_path(sequence, frame)

            # Form the tf record
            tf_example = tf.train.Example(features=tf.train.Features(feature={
                'image/height': dataset_util.int64_feature(img.shape[0]),
                'image/width': dataset_util.int64_feature(img.shape[1]),
                'image/encoded': dataset_util.bytes_feature(img_raw),
                'image/filename': dataset_util.bytes_feature(filename.encode('utf-8')),
                'image/source_id': dataset_util.bytes_feature(filename.encode('utf-8')),
                'image/format': dataset_util.bytes_feature('png'.encode('utf-8')),
            }))

            test_writer.write(tf_example.SerializeToString())

    # Close the writers
    test_writer.close()


def __generate_train_record():
    # Get the record writer ready
    train_writer = tf.python_io.TFRecordWriter('data/train.tfrecord')
    eval_writer = tf.python_io.TFRecordWriter('data/eval.tfrecord')

    print("Loading the training labels")
    # Get the sequences and loop through them
    train_examples = get_data.get_train_labels()

    # Total number of sequences
    print("Loading the images and writing to training / evaluation record")
    num_sequences = len(train_examples.sequence.unique())
    for sequence in train_examples.sequence.unique():
        print("Starting train / eval sequence:", sequence, "of", num_sequences)
        seq_df = train_examples[train_examples.sequence == sequence]
        for frame in seq_df.frame.unique():
            frame_df = seq_df[seq_df.frame == frame]
            img = get_data.load_train_image(sequence, frame)
            img_raw = img.tostring()
            filename = get_data.get_train_img_path(sequence, frame)

            x_min = []
            x_max = []
            y_min = []
            y_max = []
            classes_text = []
            classes = []

            for row in frame_df.itertuples():
                if Constants.LABEL_INDEX[row.type] != -1:
                    x_min.append(row.left)
                    x_max.append(row.right)
                    y_min.append(row.top)
                    y_max.append(row.bottom)
                    classes_text.append(row.type.encode('utf-8'))
                    classes.append(Constants.LABEL_INDEX[row.type])

            # Form the tf record
            tf_example = tf.train.Example(features=tf.train.Features(feature={
                'image/height': dataset_util.int64_feature(img.shape[0]),
                'image/width': dataset_util.int64_feature(img.shape[1]),
                'image/encoded': dataset_util.bytes_feature(img_raw),
                'image/filename': dataset_util.bytes_feature(filename.encode('utf-8')),
                'image/source_id': dataset_util.bytes_feature(filename.encode('utf-8')),
                'image/format': dataset_util.bytes_feature('png'.encode('utf-8')),
                'image/object/bbox/xmin': dataset_util.float_list_feature(x_min),
                'image/object/bbox/xmax': dataset_util.float_list_feature(x_max),
                'image/object/bbox/ymin': dataset_util.float_list_feature(y_min),
                'image/object/bbox/ymax': dataset_util.float_list_feature(y_max),
                'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
                'image/object/class/label': dataset_util.int64_list_feature(classes),
            }))

            if sequence < num_sequences * Constants.EVAL_TRAIN_SPLIT:
                train_writer.write(tf_example.SerializeToString())
            else:
                eval_writer.write(tf_example.SerializeToString())

    # Close the writers
    train_writer.close()
    eval_writer.close()


if __name__ == '__main__':
    generate_tf_record()
