# -*- coding: utf-8 -*-

import sys
import numpy as np

import chainer
import chainer.links as L
from chainer import serializers
import net

def load_image(path):
  from PIL import Image

  img = Image.open(path)

  if img.size != (28, 28):
    img.resize((28, 28))

  gray_img = img.convert('L')
  #gray_img.save('sample-gray.png')

  y=[]
  for x in img.getdata():
    y.append(255-x[0])
  y = np.asarray(y)
  y = y.astype(np.float32)
  y /= 255
  return y

def classify(model, x):
  return model.predictor(x)

def main(argv):

  if len(argv) < 4:
    print "Usage: %s [sp|mlp|cnn] model_path image_path" % argv[0]
    sys.exit()

  type = argv[1]
  model_path = argv[2]
  image_path = argv[3]

  if type == "sp":
    model = L.Classifier(net.MnistSP())
  elif type == "cnn":
    model = L.Classifier(net.MnistCNN())
  else:
    model = L.Classifier(net.MnistMLP())

  serializers.load_npz(model_path, model)

  print("input:\t%s" % image_path)

  x = load_image(image_path)
  x = chainer.Variable(np.asarray([x]))
  r = classify(model, x)

  print("output:")
  for i in range(len(r.data[0])):
    print "\t%d: %f" % (i , r.data[0][i])
  print("class:\t%d" % np.argmax(r.data[0]))


if __name__ == "__main__":

  main(sys.argv)

