if [ "$1" = "" ]; then
  echo "Usage: sh $0 model_saved_path"
  exit 1
fi

in="/home/ryorsk/SSDsegmentation/for_MTDSSD/test"
rgb=$in"/rgb/"
teach=$in"/boundingbox/"
segteach=$in"/seglabel300/"
for i in 50 60 70 80 90 100 110 120 130 140 150
do
  out="$1"/results/"$i"epoch
  python utils/consistency_evaluation.py --image "$rgb" --teach "$teach" --result "$out"/detection_txt/ --segresult "$out"/segmentation_results_gray/
done
