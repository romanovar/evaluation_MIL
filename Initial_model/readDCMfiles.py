import os
import pydicom
import numpy as np
from pathlib import Path

# ### test on a single slice
# filename = os.path.abspath("slices/000060.dcm")
# filename = os.path.dirname("C:/Users/s161590/Desktop/Data/new/LIDC-IDRI/LIDC-IDRI-0001/1.3.6.1.4.1.14519.5.2.1.6279.6001.298806137288633453246975630178/1.3.6.1.4.1.14519.5.2.1.6279.6001.179049373636438705059720603192/000001.dcm")
# ds = pydicom.dcmread(filename)  # plan dataset
# print(ds)
# pa = ds.pixel_array
# # get image position from the metadata and corresponding Z value
# slice_pos = ds[0x20, 0x32].value
# print(slice_pos[2])


# def load_dcm_file(file_name):
#     print(file_name)
#     file_path = os.path.dirname(file_name)
#     return pydicom.dcmread(file_path)


# NOTE: consider the image position and the affine transformation from the annotation to align them
def get_pixel_array_dcm(dcm):
    return dcm.pixel_array


def get_image_position_dcm(dcm):
    return dcm[0x20, 0x32].value


def find_load_dicom_files(path_to_gz="C:/Users/s161590/Desktop/Data/new/LIDC-IDRI"):
    """
    Searches recursively for dicom files starting from the common parent dir
    When dicom files is found, it is loaded and added to the final list
    :param path_to_gz: common parent directory path for all dicom files
    :return: list with all loaded dicom files found
    """
    dcm_files = []
    for src_path in Path(path_to_gz).glob('**/*.dcm'):
        dcm = pydicom.dcmread(os.path.join(os.path.dirname(src_path), os.path.basename(src_path)))
        dcm_files.append(dcm)
    return dcm_files


def order_by_slice_location(slice):
    return float(slice.SliceLocation)


def add_extra_size_matrix(mat):
    extra_x_array = np.zeros((1, mat[0, :].size))
    mat = np.append(mat, extra_x_array, axis=0)
    extra_y_array = np.zeros((mat[:, 0].size, 1))
    mat = np.append(mat, extra_y_array, axis=1)
    mat[mat.shape[0]-1, mat.shape[1]-1] = 1
    return mat


def make_4x1_vector(vec):
    nd_vec = vec * np.ones((1, 3))
    return np.append(np.transpose(nd_vec), np.ones((1, 1)), axis=0)

#
# src_path = rnf.find_nii_file()
# print("Annotation file used: "+ src_path)
# img = rnf.load_nii_file(src_path)
# img_arr= rnf.get_nii_array(img)
#
# affine = rnf.get_affine_trans(img)
# print("Affine transformation: "+ str(affine))
#
# z_an, _ = rnf.find_annotation_on_slice(img_arr, 2)
# # TODO: z_coord_annotation not curently used - delete it!
# z_ann_affine = []
# for z_coord in z_an:
#     z_coord_transl = rnf.calculate_z_after_affine(affine, z_coord)
#     z_ann_affine.append(z_coord_transl)
#     print("Z-values of annotations found: ")
#     print(z_coord_transl[2])




# dcm_files = find_load_dicom_files(test_dir)
# dcm_files.sort(key=order_by_slice_location, reverse=True)
# volume_list = []
#
# for dcm in dcm_files:
#     dcm_img_pos = get_image_position_dcm(dcm)
#     print("Z-values of slices found: ")
#     print(dcm_img_pos[2])
#
#     volume_list.append(get_pixel_array_dcm(dcm))
#     # print(get_pixel_array_dcm(dcm).shape)
#
# image = np.array(volume_list)


def load_dcm_data(test_dir):
    dcm_files = find_load_dicom_files(test_dir)
    dcm_files.sort(key=order_by_slice_location, reverse=True)
    volume_list = []
    z_dict = {}
    z_array = []
    x = 0
    y=0
    for dcm in dcm_files:
        # print(dcm)
        dcm_img_pos = np.array(get_image_position_dcm(dcm), dtype=float)
        # print(get_pixel_array_dcm(dcm).shape)

        # print("Z-values of slices found: ")
        # print(dcm_img_pos[2])
        z_dict[dcm_img_pos[2]] = 0
        z_array.append(0)
        pixel_array = get_pixel_array_dcm(dcm)
        volume_list.append(pixel_array)
        x = pixel_array.shape[0]
        y= pixel_array.shape[1]
        # print(get_pixel_array_dcm(dcm))
    result = np.reshape(volume_list, newshape=(len(dcm_files),x, y, 1))
    # print(result.shape)
    return result, z_dict



