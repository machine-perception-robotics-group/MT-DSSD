python make_list.py --mode 0 #教師画像一覧取得
python make_train_data.py #Data augmentation
python make_list.py --mode 1 #augmentationリスト一覧取得
python make_seglabel_aug_param.py #segmentation画像用のaugmentation生成(画像名だけ)
python make_list.py --mode 3 #seg画像のaugリスト一覧取得
python Train_SSD_seg.py --batchsize 12 --epoch 130 --gpu 0 --loaderjob 8
