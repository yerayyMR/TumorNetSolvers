# Copyright [2021] [HIP Applied Computer Vision Lab, Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany]
# This file has been modified from its original version.
# Modified by Zeineb Haouari on December 5, 2024

import multiprocessing
import re
from multiprocessing import Pool
from typing import Type

import numpy as np
import pandas as pd
from batchgenerators.utilities.file_and_folder_operations import *

from TumorNetSolvers.reg_nnUnet.imageio.base_reader_writer import BaseReaderWriter
from TumorNetSolvers.reg_nnUnet.imageio.reader_writer_registry import determine_reader_writer_from_dataset_json
from TumorNetSolvers.reg_nnUnet.paths import nnUNet_raw
#from reg_nnUnet.utilities.label_handling.target_handling import RegressionManager
from TumorNetSolvers.reg_nnUnet.utilities.utils import get_identifiers_from_splitted_dataset_folder, \
    get_filenames_of_train_images_and_targets


def verify_labels(label_file: str, readerclass: Type[BaseReaderWriter], expected_labels: List[int]) -> bool:
    rw = readerclass()
    seg, properties = rw.read_seg(label_file)
    found_labels = np.sort(pd.unique(seg.ravel()))  # np.unique(seg)
    unexpected_labels = [i for i in found_labels if i not in expected_labels]
    if len(found_labels) == 0 and found_labels[0] == 0:
        print('WARNING: File %s only has label 0 (which should be background). This may be intentional or not, '
              'up to you.' % label_file)
    if len(unexpected_labels) > 0:
        print("Error: Unexpected labels found in file %s.\nExpected: %s\nFound: %s" % (label_file, expected_labels,
                                                                                       found_labels))
        return False
    return True



def check_cases(image_files: List[str], label_file: str, expected_num_channels: int,
                readerclass: Type[BaseReaderWriter]) -> bool:
    rw = readerclass()
    ret = True

    images, properties_image = rw.read_images(image_files)
    segmentation, properties_seg = rw.read_seg(label_file)

    # check for nans
    print('# check for nans')
    if np.any(np.isnan(images)):
        print(f'Images contain NaN pixel values. You need to fix that by '
              f'replacing NaN values with something that makes sense for your images!\nImages:\n{image_files}')
        ret = False
    if np.any(np.isnan(segmentation)):
        print(f'Segmentation contains NaN pixel values. You need to fix that.\nSegmentation:\n{label_file}')
        ret = False

    print(' check spacings')
    # check spacings
    spacing_images = properties_image['spacing']
    spacing_seg = properties_seg['spacing']
    if not np.allclose(spacing_seg, spacing_images):
        print('Error: Spacing mismatch between segmentation and corresponding images. \nSpacing images: %s. '
              '\nSpacing seg: %s. \nImage files: %s. \nSeg file: %s\n' %
              (spacing_images, spacing_seg, image_files, label_file))
        ret = False

    # check modalities
    print('# check modalities')
    if not len(images) == expected_num_channels:
        print('Error: Unexpected number of modalities. \nExpected: %d. \nGot: %d. \nImages: %s\n'
              % (expected_num_channels, len(images), image_files))
        ret = False

    # nibabel checks
    print('nibabel checks')
    if 'nibabel_stuff' in properties_image.keys():
        # this image was read with NibabelIO
        affine_image = properties_image['nibabel_stuff']['original_affine']
        affine_seg = properties_seg['nibabel_stuff']['original_affine']
        if not np.allclose(affine_image, affine_seg):
            print('WARNING: Affine is not the same for image and seg! \nAffine image: %s \nAffine seg: %s\n'
                  'Image files: %s. \nSeg file: %s.\nThis can be a problem but doesn\'t have to be. Please run '
                  'nnUNetv2_plot_overlay_pngs to verify if everything is OK!\n'
                  % (affine_image, affine_seg, image_files, label_file))

    # sitk checks
    print('sitk checks')
    if 'sitk_stuff' in properties_image.keys():
        # this image was read with SimpleITKIO
        # spacing has already been checked, only check direction and origin
        origin_image = properties_image['sitk_stuff']['origin']
        origin_seg = properties_seg['sitk_stuff']['origin']
        if not np.allclose(origin_image, origin_seg):
            print('Warning: Origin mismatch between segmentation and corresponding images. \nOrigin images: %s. '
                  '\nOrigin seg: %s. \nImage files: %s. \nSeg file: %s\n' %
                  (origin_image, origin_seg, image_files, label_file))
        direction_image = properties_image['sitk_stuff']['direction']
        direction_seg = properties_seg['sitk_stuff']['direction']
        if not np.allclose(direction_image, direction_seg):
            print('Warning: Direction mismatch between segmentation and corresponding images. \nDirection images: %s. '
                  '\nDirection seg: %s. \nImage files: %s. \nSeg file: %s\n' %
                  (direction_image, direction_seg, image_files, label_file))

    return ret



def verify_dataset_integrity(folder: str, num_processes: int = 8) -> None:
    """
    This function performs the following checks on the dataset:
    1. Existence of dataset.json.
    2. Presence of required keys in dataset.json.
    3. Dataset structure validation.
    4. Number of training cases.
    5. Correspondence of images and labels.
    6. Consistency of images and labels.
    7. Check for NaN values.
    8. Affines, origins, and directions (for Nibabel and SimpleITK).
    """
    # Check existence of dataset.json
    dataset_json_path = join(folder, "dataset.json")
    assert isfile(dataset_json_path), f"There needs to be a dataset.json file in folder, folder={folder}"
    dataset_json = load_json(dataset_json_path)

    # Check for required keys in dataset.json
    required_keys = ['channel_names', 'numTraining', 'file_ending']
    dataset_keys = list(dataset_json.keys())
    missing_keys = [key for key in required_keys if key not in dataset_keys]
    if missing_keys:
        raise ValueError(f"Missing required keys in dataset.json: {missing_keys}")

    # Check dataset structure
    assert isdir(join(folder, "imagesTr")), f"There needs to be an imagesTr subfolder in folder, folder={folder}"
    assert isdir(join(folder, "labelsTr")), f"There needs to be a labelsTr subfolder in folder, folder={folder}"

    # Check number of training cases
    expected_num_training = dataset_json['numTraining']
    dataset = get_filenames_of_train_images_and_targets(folder, dataset_json)
    assert len(dataset) == expected_num_training, f"Expected {expected_num_training} training cases, found {len(dataset)}."

    # Check correspondence of images and labels
    if 'dataset' in dataset_json.keys():
        missing_files = [i for i in dataset.keys() if not isfile(dataset[i]['label']) or not all(isfile(img) for img in dataset[i]['images'])]
        if missing_files:
            raise FileNotFoundError(f"Missing files for identifiers: {missing_files}")

    # Determine reader/writer class
    reader_writer_class = determine_reader_writer_from_dataset_json(dataset_json, dataset[list(dataset.keys())[0]]['images'][0])

    # Consistency of images and labels
    image_files = [v['images'] for v in dataset.values()]
    label_files = [v['label'] for v in dataset.values()]
    result = []
    A= [len(dataset_json['channel_names'])] * len(dataset)
    B= [reader_writer_class] * len(dataset)
    # Sequentially process each item
    result = []
    for image_file, label_file, num_channels, reader_writer in zip(image_files, label_files, [len(dataset_json['channel_names'])] * len(dataset), [reader_writer_class] * len(dataset)):
        result.append(check_cases(image_file, label_file, num_channels, reader_writer))

    """
    with multiprocessing.get_context("spawn").Pool(num_processes) as p:
        result = p.starmap(
            check_cases,
            zip(image_files, label_files, [len(dataset_json['channel_names'])] * len(dataset), [reader_writer_class] * len(dataset))
        )"""
    if not all(result):
        raise RuntimeError('Some images have errors. Please check text output above to see which one(s) and what\'s going on.')

    # Check for NaN values
    print('\n####################')
    print('verify_dataset_integrity Done. \nIf you didn\'t see any error messages then your dataset is most likely OK!')
    print('####################\n')



if __name__ == "__main__":
    # investigate geometry issues
    example_folder = join(nnUNet_raw, 'Dataset250_COMPUTING_it0')
    num_processes = 6
    verify_dataset_integrity(example_folder, num_processes)
