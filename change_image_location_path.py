import numpy as np
from pathlib import Path
import os
folder = 'folder/to/npy/files'
img_ind0 = 'image_indices_subset_test_set_CV0_0_0.95.npy'
img_ind1 = 'image_indices_subset_test_set_CV0_1_0.95.npy'
img_ind2 = 'image_indices_subset_test_set_CV0_2_0.95.npy'
img_ind3 = 'image_indices_subset_test_set_CV0_3_0.95.npy'
img_ind4 = 'image_indices_subset_test_set_CV0_4_0.95.npy'

IMG_PATH = 'this/is/new/image/path'
npy_file0 = np.load(folder+img_ind0, allow_pickle=True)
npy_file = np.load(folder+img_ind2, allow_pickle=True)
npy_file1 = np.load(folder+img_ind1, allow_pickle=True)
npy_file3 = np.load(folder+img_ind4, allow_pickle=True)
npy_file2 = np.load(folder+img_ind3, allow_pickle=True)
assert (npy_file.all()==npy_file0.all()), "files are not the same"
assert (npy_file==npy_file1).all(), "files are not the same"
assert (npy_file2==npy_file1).all(), "filer are not the same"
assert (npy_file2==npy_file3).all(), "fitler are not the same"


def replace_all_image_paths(old_img_path_all_images, new_path):
    new_paths = []
    for img_path in old_img_path_all_images:
        img_ind = get_img_ind(img_path)
        img_new_path = find_new_img_path_from_image_index(img_ind, new_path)
        new_paths.append(str(img_new_path))
    return new_paths


def get_img_ind(old_img_path):
    return old_img_path.split('/')[-1]


def find_new_img_path_from_image_index(img_ind, new_path):
    for src_path in Path(new_path).glob('**/*.png'):
        image_ind_in_new_path = os.path.basename(src_path)
        if image_ind_in_new_path == img_ind:
            return src_path

    return None