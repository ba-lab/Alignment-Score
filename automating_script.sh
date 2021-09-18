#!/bin/bash

# Author: Bikash Shrestha, University of Missouri-St. Louis, 07-23-2021
# Description: This script runs the pipeline. It requires an MSA file and ID of the target

if [ $# -eq 0 ]
then
    echo "Input MSA was not provided"
    exit
fi

pdb_id=$2

target_dir="test/$pdb_id"

mkdir -p $target_dir

target_path1="./$target_dir/$pdb_id.npz"
target_path2="./$target_dir/$pdb_id.feats.npy"

echo $target_path1
echo $target_path2

eval "$(conda shell.bash hook)"
conda activate venv_tfv14

if [[ -f $target_path1 ]]
then
	echo " Already done!"
else
    python3 ./generate_trRosetta_prediction.py -s $1 -t1 $target_path1 -t2 $target_path2
fi

conda deactivate

conda activate venv_tfv23

python3 ./predict_lddt_score.py -f1 $target_path1 -f2 $target_path2

conda deactivate
