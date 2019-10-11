#!/bin/bash
# 运行方法: ./get_data.sh [prefix_save_path] [data_episode]
# 如：./get_data.sh ./dataset_test/expert_agent_ 200

python run.py --generate_data --expert_path=${1} --data_episode=${2}
