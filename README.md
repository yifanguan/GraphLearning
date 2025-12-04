# GraphLearning

wl_main.py: run WL test on a graph, this is used to find k distinct features before setting parameters for some other experiments

injective_main.py: run injective (expressive power) experiments

main.py: run train and test experiments, mainly for ablation

model_train_main.py: used to train a model simply without doing ablation study and store the model state

oversmoothing_main.py: over smoothing experiment (magnitude, etc)

energy_loss_main.py: energy regularization experiments

# Mup && SP (all use Adam for now)
mup_precise_layer.py: full-batch mup experiment
mup_mini_batch.py: mini-batch mup experiment

standard_experiment.py: full-batch SP experiment
standard_mini_batch.py: standard parameterization mini-batch experiment
