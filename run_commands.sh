#!/bin/bash

# Activate Conda environment
echo "Activating Conda environment: ffcv"
conda activate ffcv

# batch execution
echo "Starting batch execution..."

python Trades.py --net res18_moe_dense --dp --n_epochs 130 &   # Run in background
python AdvMoE.py --net res18_moe_dense --dp --n_epochs 130 &    # Run in background
python RT_ER.py --net res18_moe_dense --dp --n_epochs 130 --beta 1 &  # Run in background
python RT_ER.py --net res18_moe_dense --dp --n_epochs 130 --beta 3 &  # Run in background
python RT_ER.py --net res18_moe_dense --dp --n_epochs 130 --beta 9 &  # Run in background

echo "Batch execution complete!"

# Deactivate Conda Environment
conda deactivate