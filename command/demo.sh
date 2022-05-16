python src/main.py --mode demo --path . --weight weight/00000$1_weight.pt --interp_factor 1
cp demo/*.data ../hair_viewer
cd ../hair_viewer
./hairviewer epoch_00000$1.data
#./hairviewer ground_truth.data
