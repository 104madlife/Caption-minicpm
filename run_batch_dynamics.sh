#!/bin/bash

# 脚本功能：批量处理 dynamics_count 下的文件夹 3 到 10
# 运行方式：bash run_batch_dynamics.sh

# 基础路径配置
SCRIPT_PATH="/mnt/data1/bc/MiniCPM-V/batch_video_ray.py"
INPUT_BASE="/mnt/data1/data_collection_OGameData/dynamics_count"
OUTPUT_DIR="/mnt/data1/data_collection_OGameData/dynamics_count/new_caption"

# 显卡配置 (根据需要修改)
# 如果不指定，将使用 batch_video_ray.py 中的默认值 (2,3,4,5,6)
# GPU_IDS="0,1,2,3,4,5,6,7"

# 确保输出目录存在
if [ ! -d "$OUTPUT_DIR" ]; then
    echo "Creating output directory: $OUTPUT_DIR"
    mkdir -p "$OUTPUT_DIR"
fi

# 循环处理文件夹 3 到 10
for i in {3..10}; do
    INPUT_DIR="${INPUT_BASE}/${i}"

    # 检查输入目录是否存在
    if [ ! -d "$INPUT_DIR" ]; then
        echo "[WARNING] Directory not found: $INPUT_DIR, skipping..."
        continue
    fi

    echo "========================================================"
    echo "Processing Folder: $i"
    echo "Input:  $INPUT_DIR"
    echo "Output: $OUTPUT_DIR"
    echo "Time:   $(date)"
    echo "========================================================"

    # 构造命令
    CMD="/mnt/data1/bc/venvs/minicpm/bin/python $SCRIPT_PATH \
        --input_dir \"$INPUT_DIR\" \
        --output_dir \"$OUTPUT_DIR\""

    # 如果定义了 GPU_IDS，则添加到命令中
    if [ ! -z "$GPU_IDS" ]; then
        CMD="$CMD --gpu_ids \"$GPU_IDS\""
    fi

    # 执行命令
    echo "Running: $CMD"
    eval $CMD

    RET_CODE=$?
    if [ $RET_CODE -ne 0 ]; then
        echo "[ERROR] Failed to process $INPUT_DIR. Exit code: $RET_CODE"
        # 可以选择是否在这里 exit，或者继续处理下一个
        # exit 1
    else
        echo "[SUCCESS] Finished processing $INPUT_DIR"
    fi

    echo ""
    # 给一点冷却时间（可选）
    sleep 2
done

echo "All tasks completed."
