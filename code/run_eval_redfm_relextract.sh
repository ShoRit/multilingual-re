#!/bin/bash

#SBATCH --job-name=REDFM-eval-models
#SBATCH --output=/home/rdutt/slurm_outs/%j.out
#SBATCH --gres=gpu:2080Ti:1
#SBATCH --mail-type=END
#SBATCH --mail-user=rdutt@andrew.cmu.edu
#SBATCH --mem=15G
#SBATCH --array=1-6%10
#SBATCH --time=10:00:00

# ArrayTaskID DATASET MLM DEP_MODEL SRC_LANG TGT_LANG DEP SEED GNN_MODEL
# 1 redfm mbert-base stanza en en 1 15232 rgat
# 2 redfm xlmr-base stanza en es 1 15123 rgat
# 3 redfm mbert-base trankit en fr 1 15123 rgat
# 4 redfm xlmr-base stanza en fr 1 15123 rgcn
# 5 redfm xlmr-base stanza en fr 0 15123 rgcn
# 6 redfm mbert-base trankit en it 1 15232 rgcn
# 7 redfm xlmr-base stanza en it 1 98105 rgat
# 8 redfm xlmr-base trankit en it 1 98109 rgcn
# 9 redfm xlmr-base trankit en de 1 11737 rgat
# 10 redfm xlmr-base stanza en de 1 15232 rgat


config=/data/shire/projects/multilingual_re/configs/redfm_eval_missing_config.csv

DATASET=$(awk -v ArrayTaskID=$SLURM_ARRAY_TASK_ID '$1==ArrayTaskID {print $2}' $config)
MLM=$(awk -v ArrayTaskID=$SLURM_ARRAY_TASK_ID '$1==ArrayTaskID {print $3}' $config)
DEP_MODEL=$(awk -v ArrayTaskID=$SLURM_ARRAY_TASK_ID '$1==ArrayTaskID {print $4}' $config)
SRC_LANG=$(awk -v ArrayTaskID=$SLURM_ARRAY_TASK_ID '$1==ArrayTaskID {print $5}' $config)
TGT_LANG=$(awk -v ArrayTaskID=$SLURM_ARRAY_TASK_ID '$1==ArrayTaskID {print $6}' $config)
DEP=$(awk -v ArrayTaskID=$SLURM_ARRAY_TASK_ID '$1==ArrayTaskID {print $7}' $config)
SEED=$(awk -v ArrayTaskID=$SLURM_ARRAY_TASK_ID '$1==ArrayTaskID {print $8}' $config)
GNN_MODEL=$(awk -v ArrayTaskID=$SLURM_ARRAY_TASK_ID '$1==ArrayTaskID {print $9}' $config)

source activate /home/rdutt/anaconda3/envs/flow_graphs

python3 /data/shire/projects/multilingual_re/code/relextract.py --dep_model $DEP_MODEL --src_lang $SRC_LANG --tgt_lang $TGT_LANG --model_name $MLM --dep_model $DEP_MODEL --dataset $DATASET --seed $SEED --dep $DEP --gnn_model $GNN_MODEL --mode predict --batch_size 4
