#!/bin/bash
# execute explore_mini_batch_selection.py ten times
python src/main.py --multirun hydra/launcher=joblib hydra.launcher.n_jobs=6 experiment_config.repetitions_cnt=1 training_config.batch_size=16,32,64  training_config.epochs=30 model_config.kernel_initializer=he-uniform,rnd-uniform