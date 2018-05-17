from scipy import misc
from matplotlib import pyplot as plt
import numpy as np
import matplotlib as mpl

root_folder = '/home/paulo/rds_muri/paulo/hands_challenge/'
hand_object_folder = 'hand_object/'
hand_object_gt_filename = 'BoundingBox.txt'
frame_folder = 'frame/'
tracking_folder = 'tracking/1/'

image_basename = 'image_D0000000'
image_ext = '.png'

dataset_folder = hand_object_folder
gt_filename = hand_object_gt_filename


def numpy_to_plottable_rgb(numpy_img):
    img = numpy_img
    if len(numpy_img.shape) == 3:
        channel_axis = 0
        for i in numpy_img.shape:
            if i == 3 or i == 4:
                break
            channel_axis += 1
        if channel_axis == 0:
            img = numpy_img.swapaxes(0, 1)
            img = img.swapaxes(1, 2)
        elif channel_axis == 1:
            img = numpy_img.swapaxes(1, 2)
        elif channel_axis == 2:
            img = numpy_img
        else:
            return None
        img = img[:, :, 0:3]
    img = img.swapaxes(0, 1)
    return img.astype(int)

def change_res_image(image, new_res):
    image = misc.imresize(image, new_res)
    return image

def plot_boundbox(bound_box):
    mpl.rcParams['lines.linewidth'] = 5
    plt.plot([bound_box[0], bound_box[2]], [bound_box[1], bound_box[1]], 'k-')
    plt.plot([bound_box[0], bound_box[0]], [bound_box[1], bound_box[3]], 'k-')
    plt.plot([bound_box[0], bound_box[2]], [bound_box[3], bound_box[3]], 'k-')
    plt.plot([bound_box[2], bound_box[2]], [bound_box[1], bound_box[3]], 'k-')

def crop_img(image, bound_box, new_res=None):
    bound_box = list(map(int, bound_box))
    cropped_img = image[bound_box[1]:bound_box[3], bound_box[0]:bound_box[2]]
    if new_res:
        cropped_img = change_res_image(cropped_img, new_res)
    return cropped_img


imagename_to_boundbox = {}
with open(root_folder + dataset_folder + gt_filename, 'r') as f:
    idx = 0
    for line in f:
        idx += 1
        split_line = line.split()
        if len(split_line) > 0:
            imagename_to_boundbox[split_line[0]] = list(map(float, split_line[1:]))

for i in range(9):
    image_fullname = image_basename + str(i+1) + image_ext
    imagepath = root_folder + dataset_folder + 'images/' + image_fullname
    image_np = misc.imread(imagepath)
    img_RGB = numpy_to_plottable_rgb(image_np)
    img_RGB = np.swapaxes(img_RGB, 0, 1)
    print(imagename_to_boundbox[image_fullname])
    plt.imshow(img_RGB)
    plot_boundbox(imagename_to_boundbox[image_fullname])
    plt.title(image_fullname)
    plt.show()
    cropped_img = crop_img(img_RGB, imagename_to_boundbox[image_fullname], (128, 128))
    plt.imshow(cropped_img)
    plt.title(image_fullname + ' : cropped')
    plt.show()

