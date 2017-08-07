#####################################################################################################################
# Pseudo-code
#
# 1- randomly select images from labelme dataset: labelme_images_rand
# 2- segment image using slic: labelme_image_slic_segs
# 3- randomly select segment: labelme_image_slic_seg_rand
# 4- randomly select position for the insertion of segment pixels: labelme_image_slic_seg_rand_pos
# 5- insert segment pixels into target image: target_image
#
#####################################################################################################################
from os import listdir
from os.path import isfile, join
import numpy as np

from skimage.segmentation import slic
from skimage.segmentation import mark_boundaries
from skimage.util import img_as_float
import matplotlib.pyplot as plt
import argparse
import cv2

labelme_data_path = 'data/labaleme-datatset/'

def perturb_with_occl(inp_imgs, n_segs, fname_idx_start, occl_perturbed_imgs_path, final_perturbed_path, output_res):
  n_images = len(inp_imgs)
  # print(n_images)
  # 1- randomly select images from labelme dataset: labelme_images__path_rand
  labelme_images_path = [(labelme_data_path+f) for f in listdir(labelme_data_path) if isfile(join(labelme_data_path, f))]
  # print(labelme_images_path )
  # print(np.shape(labelme_images_path), " ", n_images)
  labelme_images_path_rand = np.random.choice(labelme_images_path, size=n_images, replace=False)

  # 2- segment image using slic: labelme_image_rand_segs

  inp_img_idx = 0
  perturbed_images = [];

  for labelme_image_rand_path in labelme_images_path_rand:
    print("Perturbing image #{} with occlusions - started".format(inp_img_idx+1))
    print("Perturbing image #{} with occlusions - loading image - step 1/6".format(inp_img_idx + 1))
    labelme_image_rand = cv2.imread(labelme_image_rand_path)
    dst = np.array(np.zeros(output_res))
    # print(output_res)
    # labelme_image_rand = cv2.resize(labelme_image_rand, output_res, 0, 0, cv2.INTER_CUBIC)
    # labelme_image_rand = cv2.resize(labelme_image_rand, (round(output_res[0]/2), round(output_res[1]/2)))
    labelme_image_rand = cv2.resize(labelme_image_rand, (round(output_res[0] / 1), round(output_res[1] / 1)))

    print("Perturbing image #{} with occlusions - segmentation - step 2/6".format(inp_img_idx + 1))
    labelme_image_rand_segs = slic(img_as_float(labelme_image_rand), n_segments=n_segs, sigma=5)

    # 3- randomly select segment, and extract its pixles: labelme_image_seg_rand_cropped
    print("Perturbing image #{} with occlusions - extract segment pixels - step 3/6".format(inp_img_idx + 1))
    mask = extract_mask(labelme_image_rand, labelme_image_rand_segs)
    labelme_image_seg_rand_cropped, mask_cropped = crop_image(labelme_image_rand, mask)

    # 4- randomly select position for the insertion of segment pixels: inp_img_loc_rand
    print("Perturbing image #{} with occlusions - select rand position - step 4/6".format(inp_img_idx + 1))
    inp_img_loc_rand_start, inp_img_loc_rand_end  = generate_random_loc(inp_img_idx, inp_imgs, labelme_image_seg_rand_cropped)

    # 5- insert segment pixels into target image: target_img
    print("Perturbing image #{} with occlusions - insert segment pixels into image - step 5/6".format(inp_img_idx + 1))
    target_img = cv2.imread(inp_imgs[inp_img_idx])
    # cv2.imshow("before pcclusion perturbation", target_img)
    mask_cropped_f = cv2.normalize(mask_cropped.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX)

    # cv2.imshow("alpha", mask_cropped_f)
    # cv2.imshow("overlay", labelme_image_seg_rand_cropped)
    # cv2.imshow("target image", target_img)
    # cv2.waitKey(0)

    target_img = overlay_image_alpha(target_img, labelme_image_seg_rand_cropped, inp_img_loc_rand_start, mask_cropped_f)
    # cv2.imshow("target image", target_img)
    # cv2.waitKey(0)


    # cv2.imshow("after occlusion perturbation", target_img)
    # cv2.waitKey(0)
    print("Perturbing image #{} with occlusions  - saving new image - step 6/6".format(inp_img_idx + 1))
    fname = occl_perturbed_imgs_path + "img" + '_ol_' + '{0:06d}'.format(fname_idx_start) + ".jpg"
    # cv2.imshow("Light perturbed image", pl_img)
    cv2.imwrite(fname, target_img)

    fname_final = final_perturbed_path + "img" + '{0:06d}'.format(fname_idx_start) + ".jpg"
    # cv2.imshow("Light perturbed image", pl_img)
    cv2.imwrite(fname_final, target_img)
    # perturbed_images.append(target_img)
    inp_img_idx +=1
    fname_idx_start +=1
    print("Perturbing image #{} with occlusions - ended".format(inp_img_idx + 1))

  return perturbed_images


def generate_random_loc(inp_img_idx, inp_imgs, labelme_image_seg_rand_cropped):
  labelme_image_seg_rand_cropped_sz = np.array(np.shape(labelme_image_seg_rand_cropped)[:2])
  # print(labelme_image_seg_rand_cropped_sz)
  inp_img = cv2.imread(inp_imgs[inp_img_idx])
  inp_img_sz = np.array(np.shape(inp_img)[:2])
  # print(inp_img_sz )
  labelme_image_locs_avail = np.subtract(inp_img_sz, labelme_image_seg_rand_cropped_sz)
  # print("labelme_image_rand_locs_avail ", labelme_image_locs_avail)
  inp_img_loc_rand_x = np.random.random_integers(high=labelme_image_locs_avail[1], low=0)
  inp_img_loc_rand_y = np.random.random_integers(high=labelme_image_locs_avail[0], low=0)
  inp_img_loc_start = np.array([inp_img_loc_rand_y, inp_img_loc_rand_x])
  inp_img_loc_end = np.add(inp_img_loc_start, labelme_image_seg_rand_cropped_sz)
  # print(np.shape(inp_img_loc_end))
  return inp_img_loc_start, inp_img_loc_end


def crop_image(labelme_image_rand, mask):
  dst = cv2.bitwise_and(labelme_image_rand, labelme_image_rand, mask=mask)
  # cv2.imshow("Mask", mask)
  # cv2.imshow("Applied", dst)
  bbox = cv2.boundingRect(mask)
  # cv2.rectangle(dst, (bbox[0], bbox[1]), (bbox[0] + bbox[2], bbox[1] + bbox[3]), (0, 255, 0), 2)
  # crop image
  img_corp = dst[bbox[1]:(bbox[1] + bbox[3]), bbox[0]:(bbox[0] + bbox[2])]
  mask_corp = mask[bbox[1]:(bbox[1] + bbox[3]), bbox[0]:(bbox[0] + bbox[2])]
  return img_corp, mask_corp


def extract_mask(labelme_image_rand, labelme_image_rand_segs):
  found = False
  while not found:
    labelme_image_rand_seg_rand = np.random.choice(np.unique(labelme_image_rand_segs), size=1, replace=False)
    # print(labelme_image_rand_seg_rand)
    mask = np.zeros(labelme_image_rand.shape[:2], dtype="uint8")
    mask[labelme_image_rand_segs == labelme_image_rand_seg_rand] = 255
    image_area = np.shape(labelme_image_rand)[0] * np.shape(labelme_image_rand)[1]
    mask_area = sum(sum(mask[:]))
    if 0.05 * image_area < mask_area <= 0.2 * image_area:
      found = True
  return mask

def overlay_image_alpha(img, img_overlay, pos, alpha_mask):
  """Overlay img_overlay on top of img at the position specified by
  pos and blend using alpha_mask.

  Alpha mask must contain values within the range [0, 1] and be the
  same size as img_overlay.
  """

  x, y = pos

  # Image ranges
  y1, y2 = max(0, y), min(img.shape[0], y + img_overlay.shape[0])
  x1, x2 = max(0, x), min(img.shape[1], x + img_overlay.shape[1])

  # Overlay ranges
  y1o, y2o = max(0, -y), min(img_overlay.shape[0], img.shape[0] - y)
  x1o, x2o = max(0, -x), min(img_overlay.shape[1], img.shape[1] - x)

  # Exit if nothing to do
  if y1 >= y2 or x1 >= x2 or y1o >= y2o or x1o >= x2o:
    return

  channels = img.shape[2]

  alpha = alpha_mask[y1o:y2o, x1o:x2o]
  alpha_inv = 1.0 - alpha

  for c in range(channels):
    img[y1:y2, x1:x2, c] = (alpha * img_overlay[y1o:y2o, x1o:x2o, c] +
                            alpha_inv  * img[y1:y2, x1:x2, c])
  return img


# test_images_path = ['data/experiments-images/Chrysanthemum.jpg', 'data/experiments-images/Desert.jpg']
# occl_perturbed = perturb_with_occl(test_images_path ,10)

