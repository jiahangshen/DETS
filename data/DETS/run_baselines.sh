#!/bin/bash

# ================= 配置区域 =================
# Python解释器路径 (请根据实际情况修改)
PYTHON_BIN="/root/miniconda3/envs/subset/bin/python"

# [修改] 脚本路径 (指向 GNN 版本)
SCRIPT_PATH="baseline_all_train.py"

# [修改] 采样比例
RATIO=0.4

# [修改] 设备与数据路径 (Enthalpy 任务)
DEVICE="cuda:0"
SOURCE_CSV="../enthalpy/wudily_cho.csv"
TARGET_CSV="../enthalpy/atct.csv"
PRETRAINED="" # 如果有预训练模型路径，填在这里；没有则留空

# [修改] 训练参数 (针对 Enthalpy 任务优化)
# Enthalpy 理论数据量适中(6k)，Epoch 设置
EPOCHS_P1=500
EPOCHS_FT=1000
BATCH_SIZE=32
# 绝对容忍度 (kcal/mol)
ABS_TOL=2.0

# 日志目录
LOG_DIR="logs_enthalpy_ratio_${RATIO}"
mkdir -p "$LOG_DIR"

# ================= 方法列表 =================
declare -a METHODS=(

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
echo "开始执行 Enthalpy Baseline 实验"
echo "采样比例 (Sampling Ratio): $RATIO"
echo "日志目录: $LOG_DIR"
echo "========================================================"

for method in "${METHODS[@]}"
do
    TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
    # 处理方法名中的空格
    FILE_METHOD_NAME=${method// /_}
    LOG_FILE="${LOG_DIR}/${FILE_METHOD_NAME}_${TIMESTAMP}.log"
    
    # 为每个方法创建独立的 checkpoint 目录
    SAVE_DIR="Enthalpy_checkpoints_baseline/${FILE_METHOD_NAME}"

    echo ""
    echo "--------------------------------------------------------"
    echo "[$(date '+%H:%M:%S')] 正在运行方法: $method"
    echo "日志将保存至: $LOG_FILE"
    echo "Checkpoints 将保存至: $SAVE_DIR"
    echo "--------------------------------------------------------"

    # 执行 Python 脚本
    stdbuf -oL $PYTHON_BIN $SCRIPT_PATH \
        --sampling_method "$method" \
        --sampling_ratio $RATIO \
        --source_csv "$SOURCE_CSV" \
        --target_csv "$TARGET_CSV" \
        --epochs "$EPOCHS_P1" \
        --ft_epochs "$EPOCHS_FT" \
        --batch_size $BATCH_SIZE \
        --abs_tol $ABS_TOL \
        --save_dir "$SAVE_DIR" \
        --device "$DEVICE" \
        > "$LOG_FILE" 2>&1

    # 检查退出状态
    if [ $? -eq 0 ]; then
        echo "[$(date '+%H:%M:%S')] ✅ $method 完成。"
    else
        echo "[$(date '+%H:%M:%S')] ❌ $method 失败！请检查日志: $LOG_FILE"
    fi
    
    # 稍微暂停，释放显存
    sleep 2
done

echo ""
echo "========================================================"
echo "所有 Enthalpy 任务执行完毕！"
echo "========================================================"