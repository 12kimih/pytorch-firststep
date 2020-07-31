#!/bin/bash

#SBATCH -J MNIST
#SBATCH -o MNIST_%j.log
#SBATCH -p x_corp
#SBATCH -t 11-00:00:00
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --gres=gpu:1

# >>> conda initialize >>>
# !! Contents within this block are managed by 'conda init' !!
__conda_setup="$('/opt/ohpc/pub/anaconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/opt/ohpc/pub/anaconda3/etc/profile.d/conda.sh" ]; then
        . "/opt/ohpc/pub/anaconda3/etc/profile.d/conda.sh"
    else
        export PATH="/opt/ohpc/pub/anaconda3/bin:$PATH"
    fi
fi
unset __conda_setup
# <<< conda initialize <<<

# Activate conda environment
conda activate pytorch

# Execute python script
which python3
python3 MNIST.py --cuda
