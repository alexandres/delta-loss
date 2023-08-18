#!/bin/bash

set -xe

python train_model.py problem=chess problem.model.test_iterations.low=1 problem.model.test_iterations.high=130 problem.test_batches=3 +run_id=chess
python train_model.py problem=mazes problem.model.test_iterations.low=1 problem.model.test_iterations.high=1000 problem.test_batches=3 +run_id=mazes
python train_model.py problem=prefix_sums problem.model.test_iterations.low=1 problem.model.test_iterations.high=500 problem.test_batches=3 +run_id=prefixsums

python test_model.py problem=chess problem.model.model_path=$PWD/outputs/training_default/training-chess problem.model.test_iterations.low=1 problem.model.test_iterations.high=250
python test_model.py problem=mazes problem.model.model_path=$PWD/outputs/training_default/training-mazes problem.model.test_iterations.low=1 problem.model.test_iterations.high=1000
python test_model.py problem=prefix_sums problem.model.model_path=$PWD/outputs/training_default/training-prefixsums problem.model.test_iterations.low=1 problem.model.test_iterations.high=1000