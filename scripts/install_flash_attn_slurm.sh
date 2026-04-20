#!/bin/bash
#SBATCH --job-name=flashattn-install
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --time=00:20:00
#SBATCH --chdir=/mnt/data/zixinw/arch_research
#SBATCH --output=/mnt/data/zixinw/logs/%x_%j.out
#SBATCH --error=/mnt/data/zixinw/logs/%x_%j.err

set -euo pipefail

export PATH="$HOME/.local/bin:$PATH"
export CUDA_VISIBLE_DEVICES=0,1

echo "hostname=$(hostname)"
echo "cwd=$(pwd)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"

if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi -L || true
fi

source /mnt/data/zixinw/arch_research/.venv/bin/activate

python --version
uv --version

uv pip install --python /mnt/data/zixinw/arch_research/.venv/bin/python flash-attn

python -c "import flash_attn; print('flash_attn import ok', getattr(flash_attn, '__version__', 'unknown'))"
