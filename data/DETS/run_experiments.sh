#!/bin/bash

# 定义 n 的取值范围
N_VALUES=(5 10 20 30 40 50)
ABS_PYTHON="/root/miniconda3/envs/subset/bin/python"
LOG_DIR="DETS/edl_logs"
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
    $ABS_PYTHON -u edl_full_train.py \
        --n $n \
        --tau_high 0.9 \
        --sigma_0 0.1 \
        --w_min 1e-5 \
        > "DETS/edl_logs/n=${n}.log" 2>&1

    # -----------------------------------------------------------
    # 情况 2: 带 _WIDTH 后缀
    # 参数: tau_high=0.85, sigma0=0.15, w_min=1e-4
    # -----------------------------------------------------------
    echo "[2/3] Running WIDTH n=${n}..."
    $ABS_PYTHON -u edl_full_train.py \
        --n $n \
        --tau_high 0.85 \
        --sigma_0 0.15 \
        --w_min 1e-4 \
        > "DETS/edl_logs/n=${n}_WIDTH.log" 2>&1

    # -----------------------------------------------------------
    # 情况 3: 带 _WIDTHER 后缀
    # 参数: tau_high=0.8, sigma0=0.2, w_min=1e-4
    # -----------------------------------------------------------
    echo "[3/3] Running WIDTHER n=${n}..."
    $ABS_PYTHON -u edl_full_train.py \
        --n $n \
        --tau_high 0.8 \
        --sigma_0 0.2 \
        --w_min 1e-4 \
        > "DETS/edl_logs/n=${n}_WIDTHER.log" 2>&1

    echo "================ Finished Group for n=${n} ================"
    echo ""
done

echo "All experiments finished!"
#nohup python -u DETS/edl_full_train.py --n 50 --pretrained None --tau_high 0.85 --sigma_0 0.15  --w_min 1e-4  > "DETS/edl_logs/n=50_WIDTH.log" 2>&1