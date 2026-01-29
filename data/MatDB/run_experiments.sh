#!/bin/bash
#nohup ./MatDB/run_experiments.sh > MatDB/edl_dets_logs/global_run.log 2>&1 &
# ==========================================
# Bandgap 小样本超参数搜索脚本
# ==========================================
set -e

# 1. 基础配置
PYTHON_SCRIPT="edl_dets_train.py"
BASE_LOG_DIR="edlL_dets_logs"
mkdir -p $BASE_LOG_DIR

# 2. 定义搜索空间
# Sigma (温度): 控制对"不像"样本的容忍度
# 小 sigma = 严格; 大 sigma = 宽容 (适合小样本)
SIGMAS=(0.2 0.3 0.5)

# Tau (硬门槛): 只有相似度 > tau 才算"自己人"
TAUS=(0.85 0.90 0.95)

# W_Min (保底权重): 防止小样本过拟合
# 0.1 = 温和; 0.01 = 严格
W_MINS=(0.1 0.2)

# 3. 循环搜索
for sigma in "${SIGMAS[@]}"; do
    for tau in "${TAUS[@]}"; do
        for w_min in "${W_MINS[@]}"; do
            
            # 构造实验名称
            EXP_NAME="sig${sigma}_tau${tau}_wmin${w_min}"
            SAVE_DIR="checkpoints_bandgap_tuning/${EXP_NAME}"
            LOG_FILE="${BASE_LOG_DIR}/${EXP_NAME}.log"
            
            echo "----------------------------------------------------"
            echo "Running Experiment: ${EXP_NAME}"
            echo "Config: Sigma=${sigma}, Tau=${tau}, W_Min=${w_min}"
            echo "Saving to: ${SAVE_DIR}"
            echo "Logging to: ${LOG_FILE}"
            ABS_PYTHON="/root/miniconda3/envs/subset/bin/python"
            # 运行 Python 脚本
            # 注意：你需要先修改 python 代码，使其接受命令行参数
            # (下文会提供修改后的 python 接收部分)
            nohup $ABS_PYTHON -u $PYTHON_SCRIPT \
                --sigma_0 $sigma \
                --tau_high $tau \
                --w_min $w_min \
                --num_prototypes 5 \
                --save_dir $SAVE_DIR \
                --device cuda \
                > "$LOG_FILE" 2>&1
            
            echo "Finished ${EXP_NAME}"
            
        done
    done
done

echo "All Grid Search Experiments Completed!"