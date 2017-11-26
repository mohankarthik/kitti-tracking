import os

import Constants
import pandas as pd
import skimage.io as io


def get_train_labels() -> pd:
    """
    Gets all the training labels as a pandas dataframe
    :return: The pandas dataframe containing all the training labels
    """
    _frames = []

    # Get the sequences and loop through them
    _sequences = os.listdir(Constants.TRAIN_IMAGE_PATH)
    for _sequence in _sequences:
        # Read the labels for the entire sequence
        _labels = pd.read_csv(
            filepath_or_buffer=os.path.join(Constants.TRAIN_LABEL_PATH, _sequence) + '.txt',
            delimiter=' ',
            header=None,
            names=['frame', 'track_id', 'type', 'truncated', 'occluded', 'alpha', 'left', 'top', 'right',
                   'bottom', 'height', 'width', 'length', 'x', 'y', 'z', 'rotation_y'])
        _labels['sequence'] = int(_sequence)
        _frames.append(_labels)

    return pd.concat(_frames)


def get_train_img_path(sequence: int, frame: int) -> str:
    """
    Gets the image path, given a particular sequence and frame
    :param sequence: The sequence from which the image to be loaded
    :param frame: The exact frame to be loaded
    :return: Returns the file path
    """
    return os.path.join(Constants.TRAIN_IMAGE_PATH, str(sequence).zfill(4), (str(frame).zfill(6) + '.png'))


def load_train_image(sequence: int, frame: int):
    """
    Loads an image, given a particular sequence and frame
    :param sequence: The sequence from which the image to be loaded
    :param frame: The exact frame to be loaded
    :return: Returns a numpy array that represents the image
    """
    _img = io.imread(get_train_img_path(sequence, frame))
    return _img


def get_test_sequences() -> list:
    """
    Gets the list of sequences in the test set
    :return: A list of test sequences
    """
    return os.listdir(Constants.TEST_IMAGE_PATH)


def get_test_frames(sequence: int) -> list:
    """
    Gets the list of frames in a given test sequence
    :return: A list of test sequences
    """
    return os.listdir(os.path.join(Constants.TEST_IMAGE_PATH, str(sequence).zfill(4)))


def get_test_img_path(sequence: int, frame: int) -> str:
    """
    Gets the image path, given a particular sequence and frame
    :param sequence: The sequence from which the image to be loaded
    :param frame: The exact frame to be loaded
    :return: Returns the file path
    """
    return os.path.join(Constants.TEST_IMAGE_PATH, str(sequence).zfill(4), (str(frame).zfill(6)))


def load_test_image(sequence: int, frame: int):
    """
    Loads an image, given a particular sequence and frame
    :param sequence: The sequence from which the image to be loaded
    :param frame: The exact frame to be loaded
    :return: Returns a numpy array that represents the image
    """
    _img = io.imread(get_test_img_path(sequence, frame))
    return _img


if __name__ == '__main__':
    print("Nothing to do")
