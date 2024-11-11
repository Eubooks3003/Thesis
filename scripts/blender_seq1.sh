DATASET_PATH=$1
EXP_PATH=$2

echo python active_train.py -s $DATASET_PATH -m ${EXP_PATH} --eval --method=H_reg --seed=0 --schema v20seq1_inplace --iterations 30000  --white_background --configs /home/ellina/Working/Thesis/4DGaussians/arguments/dnerf/bouncingballs.py 
python active_train.py -s $DATASET_PATH -m ${EXP_PATH} --eval --method=H_reg --seed=0 --schema v20seq1_inplace --iterations 30000  --white_background --configs /home/ellina/Working/Thesis/4DGaussians/arguments/dnerf/bouncingballs.py 