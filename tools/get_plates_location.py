#!/usr/bin/env python

# --------------------------------------------------------
# Faster R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""
Demo script showing detections in sample images.

See README.md for installation instructions before running.
"""

import sys
import _init_paths
from fast_rcnn.config import cfg
from fast_rcnn.test import im_detect
from fast_rcnn.nms_wrapper import nms
from utils.timer import Timer
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import caffe, os, cv2
import argparse
#import annnn

CLASSES = ('__background__',
		'plate')

NETS = {'vgg16': ('VGG16',
                  'VGG16_faster_rcnn_final.caffemodel'),
        'zf': ('ZF',
                  'ZF_faster_rcnn_final.caffemodel')}


class PositionDetector:
	def __init__(self):
		prototxt = '/home/mj/workspace/py-faster-rcnn/models/plates/faster_rcnn_alt_opt/faster_rcnn_test.pt'
	        caffemodel = '/home/mj/workspace/py-faster-rcnn/output/default/train/plates_faster_rcnn_final.caffemodel'
		if not os.path.isfile(caffemodel):
        		raise IOError(('{:s} not found.\nDid you run ./data/script/'
'fetch_faster_rcnn_models.sh?').format(caffemodel))

#	        if args.cpu_mode:
 #       		caffe.set_mode_cpu()
#	        els
#		print sag
        	caffe.set_mode_gpu()
        	caffe.set_device(0)
#        		cfg.GPU_ID = args.gpu_id
    		self.net = caffe.Net(prototxt, caffemodel, caffe.TEST)
		print '\n\nLoaded network {:s}'.format(caffemodel)
		# Warmup on a dummy image
	        im = 128 * np.ones((300, 500, 3), dtype=np.uint8)
	        for i in xrange(2):
			an = im_detect(self.net, im)

	def detect(self, image):
    		timer = Timer()
	        timer.tic()
#		print sa
		print image.shape
		print "!!!!!!!!!!!!!!!!!!!!!"
	        scores, boxes = im_detect(self.net, image)
	        timer.toc()
    		print ('Detection took {:.3f}s for '
	           '{:d} object proposals').format(timer.total_time, boxes.shape[0])

    		CONF_THRESH = 0.8
	        NMS_THRESH = 0.2
	       	for cls_ind, cls in enumerate(CLASSES[1:]):
		       cls_ind += 1 # because we skipped background
	               cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]
		       cls_scores = scores[:, cls_ind]
    	        dets = np.hstack((cls_boxes,
                          cls_scores[:, np.newaxis])).astype(np.float32)
       		keep = nms(dets, NMS_THRESH)
        	dets = dets[keep, :]
		inds = np.where(dets[:, -1] >= NMS_THRESH)[0]
	        if len(inds) == 0:
        		return None
	        for i in inds:
        		bbox = dets[i, :4]
        		score = dets[i, -1]
#        		cv2.rectangle(im,(bbox[0],bbox[1]),(bbox[2],bbox[3]),(0,255,0),3)
		return bbox


pose_es = PositionDetector()
image = cv2.imread("/home/mj/workspace/py-faster-rcnn/1.png")
print image.shape
pose_es.detect(image)
