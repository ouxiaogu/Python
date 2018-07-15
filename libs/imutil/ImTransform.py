# -*- coding: utf-8 -*-
"""
Created: ouxiaogu, 2018-07-15 20:20:07

Spatial domain processing for image, include:
1. intensity transformation
2. spactial domain filter

Tools includes:
1. Histogram
2. Operator like average, median, gradient, Laplace
3. Fussy set

Last Modified by: ouxiaogu
"""
import numpy as np
import cv2
import math

__all__ = ['equalizeHist', 'specifyHist']

def equalizeHist(src):
	'''
	equalize the histogram for input image

	Parameters
	----------
	src : 2D image like
		current only grayscale image supported

	Returns
	-------
	dst : 2D image like:
		histogram-equalized result for src
	'''
	if src.dtype != np.uint8:
		raise ValueError("Only single channel uint8 type image is supported, input src's dtype: {}!\n".format(repr(src.dtype)))
	if len(src.shape) != 2:
		raise ValueError("Only single channel uint8 type image is supported, input src's shape: {}!\n".format(repr(src.shape)))
	N, M = src.shape

	HBINS = 256 
	histSize = [HBINS]
	histRange = np.asarray([0, HBINS])
	hist_item = cv2.calcHist([src], [0], None, histSize, histRange.astype('uint8'))
	print(hist_item)

	hist_cdf = []
	accu = 0
	cdf_func = lambda x: min(255, max(0, uint8(math.floor((HBINS - 1) * x/(M * N) + 0.5)) ) )
	for val in hist_item:
		accu += val
		hist_cdf.append(cdf_func(accu) )

	map_func = lambda i: hist_cdf[i]
	dst = list(map(map_func, src.flatten()))
	dst.reshape(src.shape)
	return dst

if __name__ == '__main__':
	src = np.arange(12).astype('uint8').reshape((3, 4) )   

	print(equalizeHist(src) )