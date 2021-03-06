#引数にSSD学習パス(modelやoptimizerのある階層)を与えると50epoch~150epochまで推論
#modelと同じ階層のresultsに各epochごとに保存
#lossのグラフも出力
#評価まで行う
if [ "$1" = "" ] && [ "$2" = "" ] && [ "$3" = "" ]; then
  echo "Usage: sh $0 model_saved_path source_code gpu_id"
  exit 1
fi
gpu=$3
in="/home/ryorsk/SSDsegmentation/for_MTDSSD/test"
rgb=$in"/rgb/"
teach=$in"/boundingbox/"
segteach=$in"/seglabel300/"
python3 utils/show_loss_graph.py "$1"/SSD_seg_loss/ 1
mkdir "$1"/results
#for i in 50 60 70 80 90 100 110 120 130 140 150
#for i in 50 55 60 65 70 75 80 85 90 95 100 105 110 115 120 125 130 135 140 145 150
for i in 10 15 20 25 30 35 40 45 50 55 60 65 70 75 80 85 90 95 100 105 110 115 120 125 130 135 140 145 150
do
  out="$1"/results/"$i"epoch
  if [ $(($i % 4)) = 0 ]; then
    suffix=with_mining.model
  else
    suffix=without_mining.model
  fi
  mkdir $out
  python3 $2 --indir "$rgb" --outdir $out --gpu $gpu --model "$1"/model/SSD_Seg_epoch_"$i"_"$suffix"
  python3 utils/detect_evaluation.py --image "$rgb" --teach "$teach" --result "$out"/detection_txt/ --epoch $i
  python3 utils/segment_evaluation.py --teach "$segteach" --result "$out"/segmentation_results_gray/ --epoch $i
done
