import torch
import numpy as np
from src.data_handler import DataHandler
from src.experiment_run import Run

def cross_validation(
    model_class, 
    model_args, 
    x, 
    y, 
    iters,
    physics_loss, 
    physics_coef, 
    optimizer=None, 
    scheduler=None, 
    loss = torch.nn.MSELoss(),
    train_size=2000,
    data_step=20,
    physics_n=None,
    batch_size=None,
    shuffle=False,
    print_scores=True
):
    """
    cross-validation using sliding window principle for train-data-fold. training data fold size of max train_size
    """
    N = x.shape[0]

    remainder = N%train_size
    k = N // train_size if remainder < 0.5 * train_size else N // train_size + 1
    
    r2_scores = np.zeros(k)

    data_start = 0
    data_end = train_size

    for i in range(k):
        model = model_class(**model_args)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3) if optimizer is None else optimizer
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.9, patience=500,) if scheduler is None else scheduler

        run = Run(f"cv/physics_coef_{physics_coef}_{iters}_iterations/{i}")                  

        run.data_handler = DataHandler(x, y, data_start, data_end, data_step, physics_n=physics_n, batch_size=batch_size, shuffle=shuffle)                
        run.model = model

        run.optimizer = optimizer
        run.scheduler = scheduler
        run.data_loss = loss
        run.physics_loss = lambda x, y: physics_loss(x, y, loss_coef=physics_coef)

        run.train(iters)

        r2_scores[i] = run.score().detach().numpy()

        if print_scores is True:
            print(f"R2 Score on {i} validation fold: {r2_scores[i]:.3f}")

        data_start = data_end
        data_end = data_end + train_size if i < k-2 else N

    return np.mean(r2_scores)