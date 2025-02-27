DATASET_PATH=$1
EXP_PATH=$2

echo python active_train.py -s /workspace/4DGaussians/data/bouncingballs -m /workspace/output--eval --method=H_reg --seed=0 --schema v20seq1_inplace --iterations 30000  --white_background --configs /workspace/4DGaussians/arguments/dnerf/bouncingballs.py 
CUDA_LAUNCH_BLOCKING=1 python active_train.py -s /workspace/4DGaussians/data/bouncingballs -m /workspace/output --eval --method=H_reg --seed=0 --schema v20seq1_inplace --iterations 30000  --white_background --configs /workspace/4DGaussians/arguments/dnerf/bouncingballs.py 