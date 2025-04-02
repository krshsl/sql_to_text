#!/bin/bash -l
#SBATCH --output=logfile05
if [ -z "$1" ]
  then
    echo "No token supplied"
    exit 1
fi

cd /common/home/ks2025/rutgers/cs541/sql_to_text/

if [ ! -d ".venv" ]
  then
    python3.9 -m venv .venv
    source .venv/bin/activate
    pip --no-cache-dir install torch torchvision accelerate --index-url https://download.pytorch.org/whl/cu118
    pip --no-cache-dir install -r ../requirements.txt
else
    source .venv/bin/activate
fi
cd src
pgrep nvidia-smi
export 'PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:512'
export 'VLLM_USE_V1=0'
echo $PYTORCH_CUDA_ALLOC_CONF
jupyter nbconvert --execute --to notebook --inplace run_hf_models.ipynb
deactivate
