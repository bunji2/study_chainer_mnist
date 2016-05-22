#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Chainer example: train a multi-layer perceptron on MNIST

This is a minimal example to write a feed-forward net.

更新者: bunji2
更新日: 2016/05/20
更新内容: 単純パーセプトロンおよび畳み込みニューラルネットワークへの対応を追加。
"""
from __future__ import print_function
import argparse
import time

import numpy as np
import six

import chainer
from chainer import computational_graph
from chainer import cuda
import chainer.links as L
from chainer import optimizers
from chainer import serializers

import data
import net

parser = argparse.ArgumentParser(description='Chainer example: MNIST')
parser.add_argument('--initmodel', '-m', default='',
                    help='Initialize the model from given file')
parser.add_argument('--resume', '-r', default='',
                    help='Resume the optimization from snapshot')
parser.add_argument('--net', '-n', choices=('simple', 'parallel'),
                    default='simple', help='Network type')
parser.add_argument('--gpu', '-g', default=-1, type=int,
                    help='GPU ID (negative value indicates CPU)')
parser.add_argument('--epoch', '-e', default=20, type=int,
                    help='number of epochs to learn')
parser.add_argument('--unit', '-u', default=1000, type=int,
                    help='number of units')
parser.add_argument('--batchsize', '-b', type=int, default=100,
                    help='learning minibatch size')
parser.add_argument('--net2', '-t', choices=('sp', 'mlp', 'cnn'),
                    default='mlp', help='Network type2')
args = parser.parse_args()

batchsize = args.batchsize
n_epoch = args.epoch
n_units = args.unit

print('GPU: {}'.format(args.gpu))
print('# unit: {}'.format(args.unit))
print('# Minibatch-size: {}'.format(args.batchsize))
print('# epoch: {}'.format(args.epoch))
print('Network type: {}'.format(args.net))
print('Network type2: {}'.format(args.net2))
print('')

# Prepare dataset
print('load MNIST dataset')
mnist = data.load_mnist_data()
mnist['data'] = mnist['data'].astype(np.float32)
mnist['data'] /= 255
mnist['target'] = mnist['target'].astype(np.int32)

N = 60000
x_train, x_test = np.split(mnist['data'],   [N])
y_train, y_test = np.split(mnist['target'], [N])
N_test = y_test.size

type=args.net2
# Prepare model, defined in net.py
if args.net == 'simple':
    if args.net2 == 'cnn':
      model = L.Classifier(net.MnistCNN())
    elif args.net2 == 'sp':
      model = L.Classifier(net.MnistSP())
    else:
      model = L.Classifier(net.MnistMLP())
    
    if args.gpu >= 0:
        cuda.get_device(args.gpu).use()
        model.to_gpu()
    xp = np if args.gpu < 0 else cuda.cupy
elif args.net == 'parallel':
    type = 'mlp'
    cuda.check_cuda_available()
    model = L.Classifier(net.MnistMLPParallel(784, n_units, 10))
    xp = cuda.cupy

# Setup optimizer
optimizer = optimizers.Adam()
optimizer.setup(model)

# Init/Resume
if args.initmodel:
    print('Load model from', args.initmodel)
    serializers.load_npz(args.initmodel, model)
if args.resume:
    print('Load optimizer state from', args.resume)
    serializers.load_npz(args.resume, optimizer)

# Learning loop
for epoch in six.moves.range(1, n_epoch + 1):
    print('epoch', epoch)

    # training
    perm = np.random.permutation(N)
    sum_accuracy = 0
    sum_loss = 0
    start = time.time()
    for i in six.moves.range(0, N, batchsize):
        x = chainer.Variable(xp.asarray(x_train[perm[i:i + batchsize]]))
        t = chainer.Variable(xp.asarray(y_train[perm[i:i + batchsize]]))

        # Pass the loss function (Classifier defines it) and its arguments
        optimizer.update(model, x, t)

        if epoch == 1 and i == 0:
            outfile = 'graph.%s.dot' % type
            with open(outfile, 'w') as o:
                g = computational_graph.build_computational_graph(
                    (model.loss, ))
                o.write(g.dump())
            print('graph generated in %s' % outfile)

        sum_loss += float(model.loss.data) * len(t.data)
        sum_accuracy += float(model.accuracy.data) * len(t.data)
    end = time.time()
    elapsed_time = end - start
    throughput = N / elapsed_time
    print('train mean loss={}, accuracy={}, throughput={} images/sec'.format(
        sum_loss / N, sum_accuracy / N, throughput))

    # evaluation
    sum_accuracy = 0
    sum_loss = 0
    for i in six.moves.range(0, N_test, batchsize):
        x = chainer.Variable(xp.asarray(x_test[i:i + batchsize]),
                             volatile='on')
        t = chainer.Variable(xp.asarray(y_test[i:i + batchsize]),
                             volatile='on')
        loss = model(x, t)
        sum_loss += float(loss.data) * len(t.data)
        sum_accuracy += float(model.accuracy.data) * len(t.data)

    print('test  mean loss={}, accuracy={}'.format(
        sum_loss / N_test, sum_accuracy / N_test))

# Save the model and the optimizer
outfile = 'model.%s.npz' % type
print('save the model in %s' % outfile)
serializers.save_npz(outfile, model)

outfile = 'state.%s.npz' % type
print('save the optimizer in %s' % outfile)
serializers.save_npz(outfile, optimizer)
