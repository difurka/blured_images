sweep_configuration = {
        'method': 'random',
        'name': 'sweep',
        'metric': {
            'goal': 'maximize', 
            'name': 'val_acc_epoch'
            },
        'parameters': {
            'batch_size': {'values': [2, 8, 32]},
            'epochs': {'values': [10]},
            'lr': {'values': [0.003, 0.0003, 0.00003]}
        },
    }