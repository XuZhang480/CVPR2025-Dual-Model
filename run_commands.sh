#!/bin/bash

# Activate Conda environment
echo "Activating Conda environment: ffcv"
source ~/anaconda3/etc/profile.d/conda.sh  # 替换为你的 Conda 安装路径
conda activate ffcv

# batch execution
echo "Starting batch execution..."

python Trades.py --net res18_moe_dense --dp --n_epochs 130
python AdvMoE.py --net res18_moe_dense --dp --n_epochs 130
python RT_ER.py --net res18_moe_dense --dp --n_epochs 130 --beta 1
python RT_ER.py --net res18_moe_dense --dp --n_epochs 130 --beta 3
python RT_ER.py --net res18_moe_dense --dp --n_epochs 130 --beta 9

echo "Batch execution complete!"

# Deactivate Conda Environment
conda deactivate