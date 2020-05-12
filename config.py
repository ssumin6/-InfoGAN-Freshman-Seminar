# Dictionary storing network parameters.
params = {
    'batch_size': 128,# Batch size.
    'num_epochs': 100,# Number of epochs to train for.
    'learning_rateD': 2e-4,# Learning rate for D
    'learning_rateG': 1e-3,
    'beta1': 0.5,
    'beta2': 0.999,
    'save_epoch' : 5,# After how many epochs to save checkpoints and generate test output.
    'dataset' : 'MNIST'}