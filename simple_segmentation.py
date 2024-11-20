#!/usr/bin/env python3

import os
import numpy as np
import tifffile as tif

from cellpose import core, io, models, metrics


mask_filter = "_masks"
img_folder = "/path/H2B_Fusion"
path_store = "/path/segmentation"


def normalize_custom(img,
	lower=0.0,
	upper=400.0,
	background=100,
	percentile = False,
	percentile_lower = 1,
	percentile_upper = 99,
	):

	if percentile:
		x01 = np.percentile(img, percentile_lower)
		x99 = np.percentile(img, percentile_upper)
		if x99 - x01 > 1e-3:
			img = (img - x01) / (x99 - x01)
		else:
			img[:] = 0
	else:
		img = np.asarray(img)
		img = img.astype(np.float32)
		img = img - background
		img[img<0] = 0
		img = (img - lower) / (upper - lower)

	return img


model = models.CellposeModel(gpu=True, model_type="CP")

files_img = io.get_image_files(img_folder, mask_filter)


for image in files_img:
	path_img = os.path.join(img_folder, image)
	img = io.imread(path_img)

	img = normalize_custom(img)

	print("Model is evaluated")
	masks, flows, styles = model.eval(img,
									channels = [0,0],
									diameter = 25.0,
									flow_threshold = 0.4,
									cellprob_threshold = 0.0,
									stitch_threshold = 0.5,
									do_3D = False,
									normalize = False,
									)

	tif.imwrite(os.path.join(path_store, os.path.basename(path_img)), masks, compression='LZW')
