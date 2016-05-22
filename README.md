# study_chainer_mnist

Chainer の MNIST を用いた多層パーセプトロンのサンプルを改造してみる。

|スクリプト|概要|改造の有無|
|:--|:--|:--|
|data.py|MNISTのデータセットを取得するスクリプト|なし|
|net.py|ニューラルネットワークを定義するスクリプト。単純パーセプトロン MnistSP、多層パーセプトロン MnistMLP、畳み込みニューラルネット MnistCNN を定義する。|あり|
|train_mnist.py|data.py と net.py を使った学習用スクリプト。|あり|
