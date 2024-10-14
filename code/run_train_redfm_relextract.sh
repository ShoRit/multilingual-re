#!/bin/bash

#SBATCH --job-name=REDFM-infoxlm-train-models
#SBATCH --output=/home/rdutt/slurm_outs/%j.out
#SBATCH --gres=gpu:1080Ti:1
#SBATCH --mail-type=END
#SBATCH --mail-user=rdutt@andrew.cmu.edu
#SBATCH --mem=20G
#SBATCH --array=1-7
#SBATCH --time=23:00:00

# ArrayTaskID DATASET MLM DEP_MODEL SRC_LANG TGT_LANG DEP SEED GNN_MODEL
# 1 redfm mbert-base stanza it it 1 11737 rgat
# 2 redfm mbert-base trankit it it 1 11737 rgat
# 3 redfm mbert-base stanza it it 1 11737 rgcn
# 4 redfm mbert-base trankit it it 1 11737 rgcn

config=/data/shire/projects/multilingual_re/configs/redfm_train_missing_config.csv

DATASET=$(awk -v ArrayTaskID=$SLURM_ARRAY_TASK_ID '$1==ArrayTaskID {print $2}' $config)
MLM=$(awk -v ArrayTaskID=$SLURM_ARRAY_TASK_ID '$1==ArrayTaskID {print $3}' $config)
DEP_MODEL=$(awk -v ArrayTaskID=$SLURM_ARRAY_TASK_ID '$1==ArrayTaskID {print $4}' $config)
SRC_LANG=$(awk -v ArrayTaskID=$SLURM_ARRAY_TASK_ID '$1==ArrayTaskID {print $5}' $config)
TGT_LANG=$(awk -v ArrayTaskID=$SLURM_ARRAY_TASK_ID '$1==ArrayTaskID {print $6}' $config)
DEP=$(awk -v ArrayTaskID=$SLURM_ARRAY_TASK_ID '$1==ArrayTaskID {print $7}' $config)
SEED=$(awk -v ArrayTaskID=$SLURM_ARRAY_TASK_ID '$1==ArrayTaskID {print $8}' $config)
GNN_MODEL=$(awk -v ArrayTaskID=$SLURM_ARRAY_TASK_ID '$1==ArrayTaskID {print $9}' $config)

source activate /home/rdutt/anaconda3/envs/flow_graphs

python3 /data/shire/projects/multilingual_re/code/relextract.py --dep_model $DEP_MODEL --src_lang $SRC_LANG --tgt_lang $TGT_LANG --model_name $MLM --dep_model $DEP_MODEL --dataset $DATASET --seed $SEED --dep $DEP --mode train --batch_size 1 --gnn_model $GNN_MODEL
