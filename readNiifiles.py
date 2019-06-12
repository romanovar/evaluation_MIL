import nibabel as nib
import glob
import numpy as np
import matplotlib.pyplot as plt

#TODO: currently it returns only first file, maybe gather them in a list and return the list at the end
# think how to bind nifti files and dicom files
def find_nii_file( path_to_gz = "C:/Users/s161590/Desktop/Data/Output/"):
    """
    looks for nii archive files
    returns directory path where found
    """
    # path_to_gz = "C:/Users/s161590/Desktop/Data/Output/"
    for subject_no in range(1, 10):
        src_dir = path_to_gz + "000" + str(subject_no) + "a/"
        for src_path in glob.glob(src_dir + "*.nii.gz"):
            # if patient_id in src_path:
            # print(src_path)
            return src_path
    return None



def load_nii_file(search_dir):
    """"
    loads nifti file from the search_dir
    returns the file and its annotation array
    """
    img = nib.load(search_dir)
    return img

def get_nii_array(img):
    return img.get_fdata()


def get_affine_trans(img):
    return img.affine


def find_annotation_on_slice(nii_array, dim):
    """
    Find non zero matrices on a specified dimension
    :param nii_array:
    :param dim: 0, 1, 2 for x, y or z
    :return: Prints the coordinates of the annotation found
    """
    coords = []
    matrices = []
    for slice in range(0, nii_array.shape[dim]):
        if dim==0:
            non_zeroes_found = np.any(nii_array[slice, :, :])
            if non_zeroes_found:
                coords.append(slice)
                matrices.append(nii_array[slice, :, :])
        elif dim==1:
            non_zeroes_found = np.any(nii_array[:, slice, :])
            if non_zeroes_found:
                coords.append(slice)
                matrices.append(nii_array[:, slice, :])
        else:
            non_zeroes_found = np.any(nii_array[:, :, slice])
            if non_zeroes_found:
                coords.append(slice)
                matrices.append(nii_array[:, :, slice])
    return coords, matrices


def find_annotation_xy_from_z(nii_array, dim, z_coord):
    """
    :param nii_array:
    :param dim: 0 or 1 for x or y
    :param z_coord: z coordinate for the slice
    :return: list of annotation coordinates from the z-coord
    """
    coords = []
    if dim==0:
        for slice in range(0, nii_array.shape[dim]):
            non_zeroes_found = np.any(nii_array[slice, :, z_coord])
            if non_zeroes_found:
                coords.append(slice)
        return coords
    if dim==1:
        for slice in range(0, nii_array.shape[dim]):
            non_zeroes_found = np.any(nii_array[:, slice, z_coord])
            if non_zeroes_found:
                coords.append(slice)
        return coords


def find_annotation_on_x(nii_array, z_coord = None):
    if z_coord is None:
        return find_annotation_on_slice(nii_array, 0)
    else:
        return find_annotation_xy_from_z(nii_array, 0, z_coord)


def find_annotation_on_y(nii_array, z_coord=None):
    if z_coord is None:
        return find_annotation_on_slice(nii_array, 1)
    else:
        return find_annotation_xy_from_z(nii_array, 1, z_coord)


'''Visualizes the specified slice where annotation is found'''
def show_slices(slices):
    fig, axes = plt.subplots(1, len(slices))
    for i, slice in enumerate(slices):
      axes[i].imshow(slice.T, cmap="gray", origin="lower")
    plt.show()


def _rotation(point):
    return 1 if point > 0 else -1


def _translation(af_mat):
    return af_mat[:, 3]


def _scale(af_mat):
    return abs(af_mat[0, 0]), abs(af_mat[1, 1]), abs(af_mat[2, 2])


def calculate_z_after_affine(affine_mat, annotated_z_coord):
    z_vec_an = np.ones(4)
    z_vec_an[2] = annotated_z_coord
    # print(np.dot(affine_mat, z_vec_an))
    return np.array(np.dot(affine_mat, z_vec_an), dtype=float)


def load_annotation_files(z_dict):
    src_path = find_nii_file()
    # print("Annotation file used: " + src_path)
    img = load_nii_file(src_path)
    img_arr = get_nii_array(img)

    affine = get_affine_trans(img)
    # print("Affine transformation: " + str(affine))

    z_an, _ = find_annotation_on_slice(img_arr, 2)
    for z_coord in z_an:
        z_coord_transl = calculate_z_after_affine(affine, z_coord)
        if z_coord_transl[2] in z_dict.keys():
            # print("found in the dictionary ")
            z_dict[z_coord_transl[2]]=1
            # print(z_coord_transl[2])
    return z_dict








        # src_path = find_nii_file()
# print("Annotation file used: "+ src_path)
# img = load_nii_file(src_path)

# affine = get_affine_trans(img)
# print("Affine transformation: "+ str(affine))
#
# translation = _translation(affine)
# print("Translation vector: "+str(translation))
#
# scale = _scale(affine)
# print("Scale factors: "+ str(scale))
#
# rotation = _rotation(affine[0, 0]), _rotation(affine[1, 1]), _rotation(affine[2, 2])
# print("Rotation vector: "+ str(rotation))
#

# img_arr= get_nii_array(img)
# z_an,_ = find_annotation_on_slice(img_arr, 2)
# print("Z coordinates: " + str(z_an))
# x_an = find_annotation_on_x(img_arr, z_an[0])
# print("X coordinates: " + str(x_an))
# y_an = find_annotation_on_y(img_arr, z_an[0])
# print("Y-coordinates: "+ str(y_an))