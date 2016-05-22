##概要

深層学習のフレームワーク Chainer の MNIST を使ったサンプルスクリプトを試してみました。

このサンプルは MNIST という手書きの0～9の数字を分類するための多層パーセプトロン (MLP) の実装例なんですが、コード構成をじっと眺めているうち、net.py を拡張すれば、別の方式に変更できそうなことに気が付きました。

今回は、多層パーセプトロン (MLP) に加えて、単純パーセプトロン (SP)と畳み込みニューラルネットワーク (CNN) を追加しました。そして、train_mnist.py が標準出力に表示する各 epoch ごとの "accuracy" と "loss" の推移を比較してみました。

##MNISTサンプル

今回試した MNIST サンプルは[こちら](https://github.com/pfnet/chainer/tree/master/examples/mnist)にあります。

重要なスクリプトは以下の通りです。

|スクリプト|概要|
|:--|:--|
|data.py|MNISTのデータセットを読み込む。手元にデータセットがないときはネットからダウンロードする。|
|net.py|ニューラルネットワークのクラスを定義する。|
|train_mnist.py|MNISTのデータセットを net.py のニューラルネットワークで学習する。|

##使用したニューラルネットワーク

MNIST サンプルに付属の net.py に SP と CNN 用のクラスを追加しました。

```python
import chainer
import chainer.functions as F
import chainer.links as L

class MnistMLP(chainer.Chain):

    """An example of multi-layer perceptron for MNIST dataset.

    This is a very simple implementation of an MLP. You can modify this code to
    build your own neural net.

    """
    def __init__(self, n_in=784, n_units=1000, n_out=10):
        super(MnistMLP, self).__init__(
            l1=L.Linear(n_in, n_units),
            l2=L.Linear(n_units, n_units),
            l3=L.Linear(n_units, n_out),
        )

    def __call__(self, x):
        h1 = F.relu(self.l1(x))
        h2 = F.relu(self.l2(h1))
        return self.l3(h2)

# ここまでオリジナルのまま。

# ここから追加したクラス

class MnistSP(chainer.Chain):

    """An example of simple perceptron for MNIST dataset.

    """
    def __init__(self, n_in=784, n_out=10):
        super(MnistSP, self).__init__(
            l1=L.Linear(n_in, n_out),
        )

    def __call__(self, x):
        # param x --- chainer.Variable of array
        return self.l1(x)

class MnistCNN(chainer.Chain):

    """An example of convolutional neural network for MNIST dataset.
       refered page:
       http://ttlg.hateblo.jp/entry/2016/02/11/181322

    """

    def __init__(self, channel=1, c1=16, c2=32, c3=64, f1=256, \
                 f2=512, filter_size1=3, filter_size2=3, filter_size3=3):
        super(MnistCNN, self).__init__(
            conv1=L.Convolution2D(channel, c1, filter_size1),
            conv2=L.Convolution2D(c1, c2, filter_size2),
            conv3=L.Convolution2D(c2, c3, filter_size3),
            l1=L.Linear(f1, f2),
            l2=L.Linear(f2, 10)
        )

    def __call__(self, x):
        # param x --- chainer.Variable of array

        # 以下のような変換が必要
        x.data = x.data.reshape((len(x.data), 1, 28, 28))

        h = F.relu(self.conv1(x))
        h = F.max_pooling_2d(h, 2)
        h = F.relu(self.conv2(h))
        h = F.max_pooling_2d(h, 2)
        h = F.relu(self.conv3(h))
        h = F.max_pooling_2d(h, 2)
        h = F.dropout(F.relu(self.l1(h)))
        y = self.l2(h)
        return y


```

Convolution2D の入力がわかりにくかった。

学習を行うスクリプト train_mnist.py は使用したニューラルネットワークのグラフデータを dot 形式で出力してくれます。graphviz で PNG に変換しました。リンクしときます：

[MnistSP のグラフ](graph.sp.png) 

[MnistMLP のグラフ](graph.mlp.png) 

[MnistCNN のグラフ](graph.cnn.png) 

## 結果

ざっくりいうと、accuracy (高いほど良い)では SP が 0.93、MLP が 0.98、CNN が 0.99。
loss (低いほど良い)では SP が 0.26、MLP が 0.1、CNN が 0.05。

というわけで MNIST の手書き数字の認識において、今回検証した中では CNN が最も優れた方式だということを確認できました。

![accuracy](fig_accuracy.png)
![loss](fig_loss.png)

## リンク
[Chainer](http://chainer.org/)

[CNNの実装で参考にしたページ](http://ttlg.hateblo.jp/entry/2016/02/11/181322)

[コードはこちらにまとめておいてます](https://github.com/bunji2/study_chainer_mnist)