#!/bin/bash

#SBATCH --job-name=INDORE-eval-models
#SBATCH --output=/home/rdutt/slurm_outs/%j.out
#SBATCH --gres=gpu:2080Ti:1
#SBATCH --mail-type=END
#SBATCH --mail-user=rdutt@andrew.cmu.edu
#SBATCH --mem=15G
#SBATCH --array=1-450%15
#SBATCH --time=10:00:00

# ArrayTaskID DATASET MLM DEP_MODEL SRC_LANG TGT_LANG DEP SEED GNN_MODEL
# 1 indore xlmr-base stanza en en 1 15123 rgat
# 2 indore xlmr-base trankit en en 1 15123 rgat
# 3 indore xlmr-base stanza en en 1 15123 rgcn
# 4 indore xlmr-base trankit en en 1 15123 rgcn
# 5 indore xlmr-base stanza en en 0 15123 rgcn

config=/data/shire/projects/multilingual_re/configs/indore_eval_config.csv

DATASET=$(awk -v ArrayTaskID=$SLURM_ARRAY_TASK_ID '$1==ArrayTaskID {print $2}' $config)
MLM=$(awk -v ArrayTaskID=$SLURM_ARRAY_TASK_ID '$1==ArrayTaskID {print $3}' $config)
DEP_MODEL=$(awk -v ArrayTaskID=$SLURM_ARRAY_TASK_ID '$1==ArrayTaskID {print $4}' $config)
SRC_LANG=$(awk -v ArrayTaskID=$SLURM_ARRAY_TASK_ID '$1==ArrayTaskID {print $5}' $config)
TGT_LANG=$(awk -v ArrayTaskID=$SLURM_ARRAY_TASK_ID '$1==ArrayTaskID {print $6}' $config)
DEP=$(awk -v ArrayTaskID=$SLURM_ARRAY_TASK_ID '$1==ArrayTaskID {print $7}' $config)
SEED=$(awk -v ArrayTaskID=$SLURM_ARRAY_TASK_ID '$1==ArrayTaskID {print $8}' $config)
GNN_MODEL=$(awk -v ArrayTaskID=$SLURM_ARRAY_TASK_ID '$1==ArrayTaskID {print $9}' $config)

source activate /home/rdutt/anaconda3/envs/flow_graphs

python3 /data/shire/projects/multilingual_re/code/relextract.py --dep_model $DEP_MODEL --src_lang $SRC_LANG --tgt_lang $TGT_LANG --model_name $MLM --dataset $DATASET --seed $SEED --dep $DEP --gnn_model $GNN_MODEL --mode predict --batch_size 4
