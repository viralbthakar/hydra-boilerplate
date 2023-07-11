#!/bin/bash
# execute explore_mini_batch_selection.py ten times
for i in {1..10}
do
   python src/main.py --multirun hydra/launcher=joblib hydra.launcher.n_jobs=2 experiment_config.repetitions_cnt=100 training_config.batch_size=16,32,64 training_config.epochs=30 model_config.kernel_initializer=he-uniform,he-normal,glorot-uniform,glorot-normal,rnd-uniform,rnd-normal
done