python make_list.py --mode 0 #教師画像一覧取得
python make_train_data.py #Data augmentation
python make_list.py --mode 1 #augmentationリスト一覧取得
python Train_SSD.py --batchsize 12 --epoch 130 --gpu 0 --loaderjob 8
