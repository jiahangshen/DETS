#!/bin/bash

# ================= 配置区域 =================

# 1. 定义可用的 GPU ID 列表 (例如 "0 1 2 3" 或 仅 "0")
GPUS=(0 1 2) 

# 2. 定义最大并发任务数 (Total Concurrent Jobs)
# 警告：请计算好显存！
# 例如：如果显存 24G，每个任务占 3G，建议设置 MAX_JOBS=6 (留点余量)
# 如果有多张卡，这是所有卡加起来的总并发数
MAX_JOBS=4 

ABS_PYTHON="/root/miniconda3/envs/subset/bin/python"
SCRIPT_NAME="full_train.py"
LOG_DIR="SSEP/grid_search_logs"

echo "Creating log directory: ${LOG_DIR}"
mkdir -p $LOG_DIR

# ================= 超参数定义 =================
N_VALUES=(5 10 20 30 40 50)
KR_VALUES=(1.0 0.4 0.6 0.8 )
TAU_SIGMA_PAIRS=("0.8 0.2" "0.9 0.1" "0.95 0.05")
W_MIN_VALUES=(1e-4 1e-5)
LAMBDA_VALUES=(0.01 0.05 0.1)

# ================= 辅助函数与变量 =================
# 将 GPU 数组转为数组长度
NUM_GPUS=${#GPUS[@]}
# 任务计数器
COUNTER=0

# ================= 开始循环搜索 =================

for n in "${N_VALUES[@]}"; do
    for kr in "${KR_VALUES[@]}"; do
        for pair in "${TAU_SIGMA_PAIRS[@]}"; do
            set -- $pair
            tau=$1
            sigma=$2
            
            for w in "${W_MIN_VALUES[@]}"; do
                for lam in "${LAMBDA_VALUES[@]}"; do
                    
                    # 1. 检查当前后台任务数量
                    # jobs -r 列出运行中的任务，wc -l 统计行数
                    while [ $(jobs -r | wc -l) -ge $MAX_JOBS ]; do
                        sleep 5 # 如果任务满了，每 5 秒检查一次
                    done

                    # 2. 计算当前任务应该分配给哪个 GPU
                    # 取模运算实现轮询：0, 1, 0, 1 ...
                    GPU_INDEX=$((COUNTER % NUM_GPUS))
                    CURRENT_GPU=${GPUS[$GPU_INDEX]}

                    LOG_FILE="${LOG_DIR}/n${n}_kr${kr}_tau${tau}_sigma${sigma}_w${w}_lam${lam}.log"
                    
                    echo "[Running] GPU:${CURRENT_GPU} | n=${n} | kr=${kr} | tau=${tau} | w=${w} | lam=${lam}"
                    
                    # 3. 后台运行任务 (&)
                    # CUDA_VISIBLE_DEVICES 环境变量限制 Python 只看得到指定的 GPU
                    # 这样 Python 代码里的 "cuda:0" 实际上就是指派的那张卡
                    CUDA_VISIBLE_DEVICES=$CURRENT_GPU $ABS_PYTHON -u $SCRIPT_NAME \
                        --num_prototypes $n \
                        --keep_ratio $kr \
                        --tau_high $tau \
                        --sigma_0 $sigma \
                        --w_min $w \
                        --edl_lambda $lam \
                        > "$LOG_FILE" 2>&1 &
                    
                    # 4. 计数器 +1
                    COUNTER=$((COUNTER + 1))
                    
                    # 防止瞬间启动太多进程导致 CPU/IO 拥堵，稍微停顿一下
                    sleep 1 
                    
                done
            done
        done
    done
done

echo "All jobs submitted. Waiting for remaining jobs to finish..."
wait # 等待最后几个后台任务结束
echo "All grid search experiments finished!"