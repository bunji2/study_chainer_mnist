import sys
import numpy as np

import chainer
import chainer.links as L
from chainer import serializers
import net
from PIL import Image

if len(sys.argv) < 2:
  print "usage: %s image_path" % sys.argv[0]
  sys.exit()

img = Image.open(sys.argv[1])
gray_img = img.convert('L')
#gray_img.save('sample-gray.png')
d=img.getdata()
#print(d)
#print(type(d))
#print(len(d))

y=[]
for x in d:
  y.append(255-x[0])
y = np.asarray(y)
y = y.astype(np.float32)
y /= 255
#print y
#print len(y)

model = L.Classifier(net.MnistMLP())
serializers.load_npz('mlp.model', model)
yy=np.asarray([y])
#print len(yy)
x = chainer.Variable(yy)
r=model.predictor(x)
#print(r)
print("input:\t%s" % sys.argv[1])
print("output:")
for i in range(len(r.data[0])):
  print "\t%d: %f" % (i , r.data[0][i])
print("class:\t%d" % np.argmax(r.data[0]))
