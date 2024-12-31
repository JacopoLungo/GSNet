# export CUDA_VISIBLE_DEVICES=1
export DETECTRON2_DATASETS='/home/zpp2/ycy/datasets'
export RSIB_CKPT='/media/zpp2/PHDD/output/DINO-Results/vitbFT_p=8/checkpoint.pth'

RESULTS=/media/zpp2/PHDD/output/new-cat-seg-results/Final_Check_Eval_code

sh scripts/train.sh configs/vitb_384.yaml 2 $RESULTS \
SOLVER.IMS_PER_BATCH 4 \
SOLVER.MAX_ITER 31000 \
DATALOADER.NUM_WORKERS 8 \

sh scripts/eval.sh configs/vitb_384.yaml 2 $RESULTS/Iter30kEvalResults \
MODEL.WEIGHTS $RESULTS/model_0029999.pth
# MODEL.WEIGHTS $RESULTS/model_0029999.pth 


