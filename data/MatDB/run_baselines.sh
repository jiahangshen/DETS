#!/bin/bash

# ================= 配置区域 =================
# Python解释器路径
PYTHON_BIN="/root/miniconda3/envs/subset/bin/python"

# [修改] 脚本路径 (改为 MLP 版本)
SCRIPT_PATH="all_baseline_train.py"

# [修改] 采样比例 (可以根据需求调整，例如 0.2, 0.4, 0.6, 0.8)
RATIO=0.8

# [修改] 设备与数据路径 (Bandgap 任务)
DEVICE="cuda:0"
SOURCE_CSV="../bandgap/theory_features_clean.csv"
TARGET_CSV="../bandgap/exp_features_clean.csv"

# [修改] 训练参数 (针对 Bandgap 小样本任务优化)
# Bandgap 理论数据量小(几千)，预训练 epoch 需要多一点
PRE_EPOCHS=1000
# 实验数据极少(几十条)，微调 epoch 需要很多，依赖早停
FT_EPOCHS_2_1=5000 # Head Alignment
FT_EPOCHS_2_2=1000 # Full Fine-tuning

# Batch Size 设置
SOURCE_BATCH=64   # 理论数据较少，Batch 不宜过大
TARGET_BATCH=32   # 实验数据极少

# 日志目录
LOG_DIR="logs_bandgap_ratio_${RATIO}"
mkdir -p "$LOG_DIR"

# ================= 方法列表 =================
declare -a METHODS=(
    # --- 基准 ---
    "Full"
    
    # --- 静态采样 (Static) ---
    "Hard Random"
    "K-Means"
    "Herding"
    "Entropy"
    "Least Confidence"
    "EL2N-2"
    "GraNd-20"
    "DP"
    "CSIE"
    
    # --- 跨域/梯度近似 ---
    "Glister"
    "Influence"
    
    # --- 动态采样 (Dynamic) ---
    "Soft Random"
    "epsilon-greedy"
    "UCB"
    "InfoBatch"
    "MolPeg"
)

# ================= 主循环 =================

echo "========================================================"
echo "开始执行 Bandgap Baseline 实验"
echo "采样比例 (Sampling Ratio): $RATIO"
echo "日志目录: $LOG_DIR"
echo "========================================================"

for method in "${METHODS[@]}"
do
    TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
    FILE_METHOD_NAME=${method// /_}
    LOG_FILE="${LOG_DIR}/${FILE_METHOD_NAME}_${TIMESTAMP}.log"

    echo ""
    echo "--------------------------------------------------------"
    echo "[$(date '+%H:%M:%S')] 正在运行方法: $method"
    echo "日志将保存至: $LOG_FILE"
    echo "--------------------------------------------------------"

    # 执行 Python 脚本
    # 注意：参数名必须与 baseline_mlp_train.py 中的 argparse 对应
    stdbuf -oL $PYTHON_BIN $SCRIPT_PATH \
        --sampling_method "$method" \
        --sampling_ratio $RATIO \
        --theory_path "$SOURCE_CSV" \
        --exp_path "$TARGET_CSV" \
        --epochs_p1 $PRE_EPOCHS \
        --epochs_p2_1 $FT_EPOCHS_2_1 \
        --epochs_p2_2 $FT_EPOCHS_2_2 \
        --batch_size $SOURCE_BATCH \
        --device "$DEVICE" \
        > "$LOG_FILE" 2>&1

    # 检查退出状态
    if [ $? -eq 0 ]; then
        echo "[$(date '+%H:%M:%S')] ✅ $method 完成。"
    else
        echo "[$(date '+%H:%M:%S')] ❌ $method 失败！请检查日志: $LOG_FILE"
    fi
    
    sleep 2
done

echo ""
echo "========================================================"
echo "所有 Bandgap 任务执行完毕！"
echo "========================================================"