#!/bin/bash

# ================= 配置区域 =================
# Python解释器路径 (如果使用了conda环境，请确保已激活或填写完整路径)
PYTHON_BIN="/root/miniconda3/envs/subset/bin/python"

# 脚本路径
SCRIPT_PATH="./baseline_all_train.py"

# 基础参数
RATIO=0.8
DEVICE="cuda:0"
SOURCE_CSV="./solute-solvent/qm.csv"
TARGET_CSV="./solute-solvent/exp_data.csv"

# 训练参数 (根据需要调整)
PRE_EPOCHS=100
FT_EPOCHS=200
SOURCE_BATCH=4096
TARGET_BATCH=256

# 日志目录
LOG_DIR="logs_baseline_ratio_${RATIO}"
mkdir -p "$LOG_DIR"

# ================= 方法列表 =================
# 你可以在这里注释掉不想跑的方法
declare -a METHODS=(
    # --- 基准 ---
    #"Full"
    
    # --- 静态采样 (Static) ---
    #"Hard Random"
    #"K-Means"
    #"Herding"
    #"Entropy"
    "Least Confidence"
    "EL2N-2"
    "GraNd-20"
    "DP"
    "CSIE"
    
    # --- 跨域/梯度近似 (较慢) ---
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
echo "开始顺序执行 Baseline 实验"
echo "采样比例 (Sampling Ratio): $RATIO"
echo "日志目录: $LOG_DIR"
echo "========================================================"

for method in "${METHODS[@]}"
do
    # 生成带时间戳的日志文件名
    TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
    # 将方法名中的空格替换为下划线 (用于文件名)
    FILE_METHOD_NAME=${method// /_}
    LOG_FILE="${LOG_DIR}/${FILE_METHOD_NAME}_${TIMESTAMP}.log"

    echo ""
    echo "--------------------------------------------------------"
    echo "[$(date '+%H:%M:%S')] 正在运行方法: $method"
    echo "日志将保存至: $LOG_FILE"
    echo "--------------------------------------------------------"

    # 执行 Python 脚本
    # 使用 stdbuf -oL 确保日志实时写入文件，而不是缓存
    stdbuf -oL $PYTHON_BIN $SCRIPT_PATH \
        --sampling_method "$method" \
        --sampling_ratio $RATIO \
        --source_csv "$SOURCE_CSV" \
        --target_csv "$TARGET_CSV" \
        --pre_epochs $PRE_EPOCHS \
        --ft_epochs $FT_EPOCHS \
        --source_batch_size $SOURCE_BATCH \
        --batch_size $TARGET_BATCH \
        --device "$DEVICE" \
        > "$LOG_FILE" 2>&1

    # 检查退出状态
    if [ $? -eq 0 ]; then
        echo "[$(date '+%H:%M:%S')] ✅ $method 完成。"
    else
        echo "[$(date '+%H:%M:%S')] ❌ $method 失败！请检查日志: $LOG_FILE"
        # 如果希望出错后继续跑下一个，请保留下面这一行；如果希望出错即停止，请取消注释 'exit 1'
        # exit 1
    fi
    
    # 可选：每跑完一个休息几秒，让显卡降温或释放显存
    sleep 3
done

echo ""
echo "========================================================"
echo "所有任务执行完毕！"
echo "========================================================"