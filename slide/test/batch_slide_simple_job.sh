#!/bin/bash

#SBATCH --job-name=wsi_emb                       # Job name
#SBATCH --output=/cluster/CBIO/data1/mblons/tipit/slurm/log/slurm-%x_%j.log     # Output log
#SBATCH --error=/cluster/CBIO/data1/mblons/tipit/slurm/log/slurm-%x_%j.err      # Error log
#SBATCH --mem 16000                              # Job memory request
#SBATCH -p cbio-gpu                              # Name of the partition to use
#SBATCH --nodelist=node006                       # Choose node
#SBATCH --exclude=node009                        # Do not Use
#SBATCH --cpus-per-task=4                        # CPU cores per process (default 1, typically 4 or 5 - do not use more to let space for other people)

CONFIG="./config_default.yaml"
SRC="/cluster/CBIO/data1/mblons/tipit/data/pathology/image/hes/wsi"
DST="/cluster/CBIO/data1/mblons/tipit/data/pathology/image/hes/features"

python test_batch_slide_tile_feat_extraction.py --wsi_dir $SRC --job_dir $DST --config $CONFIG --clock
wait