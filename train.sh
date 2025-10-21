#!/bin/bash

# Create directories
mkdir -p logs
mkdir -p csv

# 固定参数
MODE="isnet"
FEATURE_MODE="facial-only"
SELECTED_FEATURE="all"
TRANSLATOR_MODE="ED-LSTM"
NEU=-1
EDGE=0
LR=0.01
BATCH_SIZE=32
NUM_LAYER=6
NUM_HEAD=4
HIDDEN_DIM=256
IS_DYNOISE=4
IS_VISUALBC=False
SLE_NOISE=True
FLE_NOISE=True
AVE_METHOD="temp_attn_mean"
MODEL="transformer"
PREPROCESSING="subtract"

# 日志和输出文件
RESULTS_CSV="csv/single_run.csv"
LOG_FILE="logs/f&s.log"

# 写入CSV表头
echo "mode,translator_mode,model,feature_mode,selected_feature,preprocessing,ave_method,sle_noise,fle_noise,is_dynoise,is_visualbc,neu,edge,hidden_dim,batch_size,lr,num_layer,num_head,best_val_mse,best_ccc,best_Pearson_Correlation,best_Pearson_p-value,best_Kendall_Correlation,best_Kendall_p-value" > $RESULTS_CSV

# 打印参数信息
echo "Running fixed experiment with parameters:"
echo "Mode: $MODE"
echo "Feature Mode: $FEATURE_MODE"
echo "Selected Feature: $SELECTED_FEATURE"
echo "Translator Mode: $TRANSLATOR_MODE"
echo "Neu: $NEU"
echo "Edge: $EDGE"
echo "Learning Rate: $LR"
echo "Batch Size: $BATCH_SIZE"
echo "Num Layers: $NUM_LAYER"
echo "Num Heads: $NUM_HEAD"
echo "Hidden Dim: $HIDDEN_DIM"
echo "Is_dynoise: $IS_DYNOISE"
echo "Is_visualbc: $IS_VISUALBC"
echo "SLE Noise: $SLE_NOISE"
echo "FLE Noise: $FLE_NOISE"
echo "AVE Method: $AVE_METHOD"
echo "Log: $LOG_FILE"

# 运行Python脚本
python train.py \
    --mode $MODE \
    --feature_mode $FEATURE_MODE \
    --selected_feature $SELECTED_FEATURE \
    --translator_mode $TRANSLATOR_MODE \
    --neu $NEU \
    --edge $EDGE \
    --lr $LR \
    --batch_size $BATCH_SIZE \
    --num_layers $NUM_LAYER \
    --num_heads $NUM_HEAD \
    --hidden_dim $HIDDEN_DIM \
    --is_dynoise $IS_DYNOISE \
    --is_visualbc $IS_VISUALBC \
    --sle_noise $SLE_NOISE \
    --fle_noise $FLE_NOISE \
    --ave_method $AVE_METHOD \
    --epochs 100 \
    --early_stop 21 \
    --seed 1 > $LOG_FILE 2>&1

# 提取结果
BEST_MSE=$(grep "final_best_val_mse:" $LOG_FILE | awk '{print $2}')
BEST_CCC=$(grep "final_best_ccc:" $LOG_FILE | awk '{print $2}')
BEST_PEARSON=$(grep "final_best_Pearson_Correlation:" $LOG_FILE | awk '{print $2}')
BEST_PEARSON_P=$(grep "final_best_Pearson_p-value:" $LOG_FILE | awk '{print $2}')
BEST_KENDALL=$(grep "final_best_Kendall_Correlation:" $LOG_FILE | awk '{print $2}')
BEST_KENDALL_P=$(grep "final_best_Kendall_p-value:" $LOG_FILE | awk '{print $2}')

# 写入CSV
echo "$MODE,$TRANSLATOR_MODE,$MODEL,$FEATURE_MODE,$SELECTED_FEATURE,$PREPROCESSING,$AVE_METHOD,$SLE_NOISE,$FLE_NOISE,$IS_DYNOISE,$IS_VISUALBC,$NEU,$EDGE,$HIDDEN_DIM,$BATCH_SIZE,$LR,$NUM_LAYER,$NUM_HEAD,$BEST_MSE,$BEST_CCC,$BEST_PEARSON,$BEST_PEARSON_P,$BEST_KENDALL,$BEST_KENDALL_P" >> $RESULTS_CSV

echo "Single experiment completed. Results saved to $RESULTS_CSV"
