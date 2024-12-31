# export CUDA_VISIBLE_DEVICES=1
export DETECTRON2_DATASETS='/home/zpp2/ycy/datasets'
export RSIB_CKPT='/media/zpp2/PHDD/output/DINO-Results/vitbFT_p=8/checkpoint.pth'

RESULTS=/media/zpp2/PHDD/output/new-cat-seg-results/Final_dual_Seed3097

sh scripts/train.sh configs/vitb_384.yaml 2 $RESULTS \
TEST.EVAL_PERIOD 0 \
SOLVER.IMS_PER_BATCH 4 \
SOLVER.MAX_ITER 31000 \
SEED 30975509 \
# DATALOADER.NUM_WORKERS 8 \
# From Ver 0.7, we modify the seg_head

sh scripts/eval.sh configs/vitb_384.yaml 2 $RESULTS/Iter30kEvalResults \
MODEL.WEIGHTS $RESULTS/model_0029999.pth 
# MODEL.WEIGHTS $RESULTS/model_0029999.pth 


