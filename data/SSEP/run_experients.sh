#!/bin/bash

# ================= 配置区域 =================
# Python解释器绝对路径
ABS_PYTHON="/root/miniconda3/envs/subset/bin/python"
# 你的训练脚本文件名 (请确保文件名正确，例如 full_train.py)
SCRIPT_NAME="full_train.py"
# 日志保存目录
LOG_DIR="SSEP/grid_search_logs"

# 创建日志目录
echo "Creating log directory: ${LOG_DIR}"
mkdir -p $LOG_DIR

# ================= 超参数定义 =================

# 1. num_prototypes (N)
N_VALUES=(5 10 20 30 40 50)

# 2. keep_ratio (KR)
# 注意：请确保 Python 代码中该参数类型已改为 float
KR_VALUES=(1.0 0.6 0.8 )

# 3. tau_high 和 sigma_0 的组合 (Tau_Sigma)
# 格式为 "tau_high sigma_0" 的字符串
TAU_SIGMA_PAIRS=(
    "0.8 0.2" 
    "0.9 0.1" 
    "0.95 0.05"
)

# 4. w_min (W)
W_MIN_VALUES=(1e-4 1e-5)

# 5. edl_lambda (Lam)
LAMBDA_VALUES=(0.01 0.05 0.1)

# ================= 开始循环搜索 =================

# 第一层循环：Prototypes (N)
for n in "${N_VALUES[@]}"; do
    echo "================ Starting Group for Num Prototypes (n) = ${n} ================"

    # 第二层循环：Keep Ratio (KR)
    for kr in "${KR_VALUES[@]}"; do
        
        # 第三层循环：Tau & Sigma 组合
        for pair in "${TAU_SIGMA_PAIRS[@]}"; do
            # 解析组合字符串
            set -- $pair
            tau=$1
            sigma=$2
            
            # 第四层循环：w_min (W)
            for w in "${W_MIN_VALUES[@]}"; do
                
                # 第五层循环：edl_lambda (Lam)
                for lam in "${LAMBDA_VALUES[@]}"; do
                    
                    # 构建日志文件名，包含所有参数以便区分
                    LOG_FILE="${LOG_DIR}/n${n}_kr${kr}_tau${tau}_sigma${sigma}_w${w}_lam${lam}.log"
                    
                    echo "[Running] n=${n} | kr=${kr} | tau=${tau}/sig=${sigma} | w=${w} | lam=${lam}"
                    echo "   -> Log: ${LOG_FILE}"

                    # 执行 Python 命令
                    $ABS_PYTHON -u $SCRIPT_NAME \
                        --num_prototypes $n \
                        --keep_ratio $kr \
                        --tau_high $tau \
                        --sigma_0 $sigma \
                        --w_min $w \
                        --edl_lambda $lam \
                        > "$LOG_FILE" 2>&1
                    
                done
            done
        done
    done
    
    echo "================ Finished Group for n=${n} ================"
    echo ""
done

echo "All grid search experiments finished!"