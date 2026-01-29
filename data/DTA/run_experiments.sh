#!/bin/bash

# 定义 n 的取值范围
N_VALUES=(3 4 5 6 7 8 9 10)
ABS_PYTHON="/root/miniconda3/envs/subset/bin/python"
LOG_DIR="DTA/icl_dets_logs"
echo "Creating log directory: ${LOG_DIR}"
mkdir -p $LOG_DIR
# 循环遍历每一个 n 值
for n in "${N_VALUES[@]}"; do
    echo "================ Starting Group for n=${n} ================"

    # -----------------------------------------------------------
    # 情况 1: 不带后缀 (Standard)
    # 参数: tau_high=0.9, sigma0=0.1, w_min=1e-5
    # -----------------------------------------------------------
    echo "[1/3] Running standard n=${n}..."
    $ABS_PYTHON -u icl_dets_train.py \
        --num_prototypes $n \
        > "DTA/icl_dets_logs/n=${n}.log" 2>&1

    echo "================ Finished Group for n=${n} ================"
    echo ""
done

echo "All experiments finished!"
#nohup python -u DETS/edl_full_train.py --n 50 --pretrained None --tau_high 0.85 --sigma_0 0.15  --w_min 1e-4  > "DETS/edl_logs/n=50_WIDTH.log" 2>&1
#nohup ./DTA/run_experiments.sh > DTA/edl_dets_logs/global_run.log 2>&1 &