# MT-DSSD
[Team MC^2 : ARC2017 RGB-D Dataset](http://mprg.jp/research/arc_dataset_2017_j)向けのサンプルコードです．

- 2018/08/02 [MIRU2018で発表するポスター(33.1MB)](http://www.mprg.cs.chubu.ac.jp/~ryorsk/share/MIRU2018Poster_small.pdf)を公開中です．
- 2018/08/08 ポスターに間違いが2件ありました．お詫びして訂正致します．
  - 提案手法/Loss function [L(y,c,g,s)] -> [L(y,c,l,s)]
  - 評価実験/評価サンプルの枚数 [20枚] -> [200枚]

## Contents
- SSD[1]
- MultiTask DSSD(MT-DSSD)[2, 3]

# Requirement
- Python 2.x (Recommended version >= 2.7.11)
- OpenCV 2.x
- Chainer 1.x (Recommended version == 1.7.0)
- numpy (Recommended version >= 1.10)

# Usage

## SSD / MT-DSSD共通の手順

### 1. データセットのダウンロード
下記のURLからデータセット(png版)をダウンロードしてください．

- Web: http://mprg.jp/research/arc_dataset_2017_j
- Direct URL: http://www.mprg.cs.chubu.ac.jp/ARC2017/ARCdataset_png.zip

ダウンロード完了後，unzipして適当な場所に置いて下さい．

### 2. データセットのパス設定
`common_params.py`の下記の部分をデータセットのパスに書き換えて下さい．
```
# 学習画像のディレクトリ
images_dir = '<Your dataset path>'
```

### 3. データセットのディレクトリ追加
データセットのディレクトリ内に，必要な空ディレクトリを作成します．
`mkdirs.sh`を実行することで作成できます．引数にデータセットのパスを与えてください．

```
sh mkdirs.sh <Your Dataset Path>
```

### 4. 教師データ一覧の取得
`make_list.py`に，引数`--mode 0`を与えて実行します．
```
python make_list.py --mode 0
```
`img_name_list.txt`に教師データのファイル名一覧が保存されます．

### 5. Data augmentation
本コードは学習前に予めData augmentationを行い，変換パラメータを保存します．
`make_train_data.py`を実行して，Data augmentationを開始します．
```
python make_train_data.py
```

### 6. Data augmentation後の教師データ一覧の取得
`make_list.py`に，引数`--mode 1`を与えて実行します．
```
python make_list.py --mode 1
```
`augimg_name_list.txt`にData augmentation後の教師データのファイル名一覧が保存されます．

## SSDの手順
※`Train_SSD.sh`を実行することで，共通手順4, 5, 6と学習を連続して行うことができます．

### 7. 学習
`Train_SSD.py`を実行して下さい．

```
python Train_SSD.py --batchsize 12 --epoch 130 --gpu 0 --loaderjob 8
```

下記の引数を与えることができます．
- `--batchsize` (default = 8)
- `--epoch` (default = 80)
- `--gpu` (default = -1) : GPU ID．マイナス値にするとCPUを用いる．
- `--loaderjob` (default = 4) : 並列実行数．


### 8. テスト
`Test_SSD.py`の下記の部分を書き換えて下さい．

```
serializers.load_npz('<Your Trained Model Path>', ssd_model)
```
`<Your Trained Model Path>`に学習済みモデルのパスを入力します．

続いて，`Test_SSD.py`を実行して下さい．指定されたパスにある指定された拡張子の画像すべてを用いてテストを行います．

```
python Test_SSD.py --dir '<Your Dataset Path>/test_known/rgb' --out './out' --type '.png' --gpu 0
```

下記の引数を与えることができます．
- `--dir` (default='./') : テスト用画像のあるパス
- `--out` (default='./') : 検出結果の出力先
- `--type` (default='.jpg') : テスト用画像の拡張子
- `--gpu` (default = -1) : GPU ID．マイナス値にするとCPUを用いる．


## MT-DSSDの手順
※`Train_SSD.sh`を実行することで，共通手順4, 5, 6とMT-DSSDの手順8, 9, 学習を連続して行うことができます．6, 7は実行されませんので，予め行ってください．

### 6. セグメンテーション用画像の準備
セグメンテーション画像を学習用に変換する必要があります．
`convertC2G_fast.py`の下記の部分について，
`<Yout dataset path>`をデータセットのパスに書き換えて下さい．

```
# Original dataset dir (Source)
original_dataset = "<Yout dataset path>"
```

続いて，`convertC2G_fast.py`を実行してください．
```
python convertC2G_fast.py
```


### 7. セグメンテーション画像のパス設定
`make_seglabel_aug_param.py`の下記の部分について，
`<Yout dataset path>`をデータセットのパスに書き換えて下さい．
```
#画像リスト
inDir = "<Yout dataset path>/train/seglabel/"
#ラベル出力先
outDir = "<Yout dataset path>/train/seglabel_aug_param/"
```

### 8. セグメンテーション画像のファイル名取得
Data augmentation後の教師データとセグメンテーション教師データを紐付けるファイルを作成するために，`make_seglabel_aug_param.py`を実行します．
```
python make_seglabel_aug_param.py
```

### 9. セグメンテーション画像ファイル名の一覧取得
`make_list.py`に，引数`--mode 3`を与えて実行します．
```
python make_list.py --mode 3
```
`seglabel_name_list.txt`に手順8で作成したファイルの名前一覧が保存されます．

### 10. 学習
```
python Train_SSD_seg.py --batchsize 12 --epoch 130 --gpu 0 --loaderjob 8
```

### 11. テスト
`Test_SSD_seg_fast.py`の下記の部分を書き換えて下さい．

```
# 学習モデルのパス
MODEL_PATH = "`<Your Trained Model Path>"
```
`<Your Trained Model Path>`に学習済みモデルのパスを入力します．

続いて，`Test_SSD_seg_fast.py`を実行して下さい．指定されたパスにある指定された拡張子の画像すべてを用いてテストを行います．

```
python Test_SSD_seg_fast.py --dir '<Your Dataset Path>/test_known/rgb' --out './out' --type '.png' --gpu 0
```
下記の引数を与えることができます．
- `--webcam` (default = -1) : 1を指定するとWebCamから得られる画像を用いてリアルタイム検出を行います．
- `--indir` : テスト用画像のあるパス
- `--outdir` (default = './out/') : 検出結果の出力先
- `--gpu` (default = -1) : GPU ID．マイナス値にするとCPUを用いる．


# Reference
- [1] W. Liu, et al., “SSD: Single Shot MultiBox Detector”, ECCV, pp. 21–37, 2016.
- [2] C. Fu, et al., “DSSD: Deconvolutional Single Shot Detector”, arXiv preprint arXiv:1701.06659, 2017.
- [3] 荒木, et al., “マルチタスク学習を導入したDeconvolutional Single Shot Detectorによる物体検出とセグメンテーションの高精度化”, MIRU, 2018.
