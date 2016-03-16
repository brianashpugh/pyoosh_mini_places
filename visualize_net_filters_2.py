#!/usr/bin/env python

from __future__ import division

import argparse
import numpy as np
import os
import tempfile
import time
import numpy as np
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(
    description='Visualize the filters on the first layer of the net.')
parser.add_argument('--snapshot_path', default='./snapshot',
    help='Path to the .caffemodel of the net')
args = parser.parse_args()

# disable most Caffe logging (unless env var $GLOG_minloglevel is already set)
key = 'GLOG_minloglevel'
if not os.environ.get(key, ''):
    os.environ[key] = '3'

import caffe
from caffe.proto import caffe_pb2
from caffe import layers as L
from caffe import params as P

if args.gpu >= 0:
    caffe.set_mode_gpu()
    caffe.set_device(args.gpu)
else:
    caffe.set_mode_cpu()

def vis_filters(filters):
    """Take an array of shape (n, height, width) or (n, height, width, 3)
       and visualize each (height, width) thing in a grid of size approx. sqrt(n) by sqrt(n)"""
    
    # normalize data for display
    data = (data - data.min()) / (data.max() - data.min())
    
    # force the number of filters to be square
    n = int(np.ceil(np.sqrt(data.shape[0])))
    padding = (((0, n ** 2 - data.shape[0]),
               (0, 1), (0, 1))                 # add some space between filters
               + ((0, 0),) * (data.ndim - 3))  # don't pad the last dimension (if there is one)
    data = np.pad(data, padding, mode='constant', constant_values=1)  # pad with ones (white)
    
    # tile the filters into an image
    data = data.reshape((n, n) + data.shape[1:]).transpose((0, 2, 1, 3) + tuple(range(4, data.ndim + 1)))
    data = data.reshape((n * data.shape[1], n * data.shape[3]) + data.shape[4:])
    
    plt.imshow(data); plt.axis('off')

if __name__ == '__main__':
    print 'Loading net...\n'
    test_net_file = miniplaces_net(split_file, train=False, with_labels=False)
    weights_file = args.snapshot_path
    model = caffe_pb2.NetParameter()
    model.ParseFromString(
        open(snapshot_path))
    conv1_layer = model.layers[2]
    filters = np.asarray(conv1_layer.blobs[0].data).reshape(
        conv1_layer.blobs[0].num, conv1_layer.blobs[0].height,
        conv1_layer.blobs[0].width, conv1_layer.blobs[0].channels)
    vis_filters(filters)


