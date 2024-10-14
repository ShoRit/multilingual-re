#!/bin/bash

#SBATCH --job-name=REDFM-data-creation
#SBATCH --output=/home/rdutt/slurm_outs/%j.out
#SBATCH --mail-type=END
#SBATCH --mail-user=rdutt@andrew.cmu.edu
#SBATCH --mem=10G
#SBATCH --array=1-5
#SBATCH --time=23:00:00


# ArrayTaskID MLM DEP_MODEL MAX_LEN LANG
# 1 mbert-base stanza 512 hi
# 2 mbert-base trankit 512 hi
# 3 mbert-base stanza 512 en


config=/data/shire/projects/multilingual_re/configs/redfm_prompt_creation_config.csv

MLM=$(awk -v ArrayTaskID=$SLURM_ARRAY_TASK_ID '$1==ArrayTaskID {print $2}' $config)
DEP_MODEL=$(awk -v ArrayTaskID=$SLURM_ARRAY_TASK_ID '$1==ArrayTaskID {print $3}' $config)
MAX_LEN=$(awk -v ArrayTaskID=$SLURM_ARRAY_TASK_ID '$1==ArrayTaskID {print $4}' $config)
LANG=$(awk -v ArrayTaskID=$SLURM_ARRAY_TASK_ID '$1==ArrayTaskID {print $5}' $config)

source activate /home/rdutt/anaconda3/envs/flow_graphs

python3 /data/shire/projects/multilingual_re/code/preprocess.py --dep_model $DEP_MODEL --max_seq_len $MAX_LEN --ml_model $MLM --step create_redfm_prompt --lang $LANG
