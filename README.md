# study_chainer_mnist

## はじめに
Chainer の MNIST を用いた多層パーセプトロンのサンプルを改造してみる。

|スクリプト|概要|改造の有無|
|:--|:--|:--|
|data.py|MNISTのデータセットを取得するスクリプト|なし|
|net.py|ニューラルネットワークを定義するスクリプト。単純パーセプトロン MnistSP、多層パーセプトロン MnistMLP、畳み込みニューラルネット MnistCNN を定義する。|あり|
|train_mnist.py|data.py と net.py を使った学習用スクリプト。|あり|


## train_mnist.py

- 処理内容

MNIST のデータセットを使って学習する。

- コマンドライン

```
tran_mnist.py --net2 [sp|mlp|cnn]
```

--net2 オプションを追加。

|値|概要|
|:--|:--|
|sp|単純パーセプトロン|
|mlp|多層パーセプトロン|
|cnn|畳み込みニューラルネット|

- 実行例
```
\# train_mnist.py --net2 mlp
```

## classify.py

- 処理内容

0～9 の手書き数字の画像を分類する。

- コマンドライン

```
Usage: classify.py [sp|mlp|cnn] model_path image_path
```

第一引数：使用するニューラルネット
第二引数：train_mnist.py で作成したモデルデータのパス
第三引数：分類対象となる手書き数字の画像データのパス

|第一引数の値|概要|
|:--|:--|
|sp|単純パーセプトロン|
|mlp|多層パーセプトロン|
|cnn|畳み込みニューラルネット|

- 実行例

```
\# python classify.py cnn model.cnn.npz number/four.png
input:  number/four.png
output:
        0: -14.180834
        1: 2.686594
        2: -2.928259
        3: -28.385714
        4: 12.872291
        5: -14.222157
        6: -5.834879
        7: -9.659436
        8: -16.648321
        9: -11.859789
class:  4
```

CNN で入力画像を分類した結果が「4」であることを示す。
