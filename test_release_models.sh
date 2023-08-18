#!/bin/bash

set -xe

SAVEDIR=$PWD/releasemodels

if [ ! -e ${SAVEDIR} ]; then
    echo "Downloading models to ${SAVEDIR}"
    python -c "import huggingface_hub; huggingface_hub.snapshot_download(repo_id='alexsalle/delta-loss-models', repo_type='dataset', local_dir='${SAVEDIR}')"
fi

python test_model.py problem=chess problem.model.model_path=${SAVEDIR}/training-jingly-Melisse problem.model.test_iterations.low=1 problem.model.test_iterations.high=250
python test_model.py problem=mazes problem.model.model_path=${SAVEDIR}/training-plotless-Shanise problem.model.test_iterations.low=1 problem.model.test_iterations.high=1000
python test_model.py problem=prefix_sums problem.model.model_path=${SAVEDIR}/training-sparid-Adrena problem.model.test_iterations.low=1 problem.model.test_iterations.high=1000