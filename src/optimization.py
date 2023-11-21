import torch
from torch import nn
from torch.utils.data import DataLoader

from tqdm import tqdm

import matplotlib.pyplot as plt
from src.utils import plot_result


def get_data_for_training(x, y, end=40, step=4, batch_size=1000, shuffle=True):
    x_data = x[0:end:step]
    y_data = y[0:end:step]
    train_dataloader = DataLoader(
        list(zip(x_data, y_data)), batch_size=batch_size, shuffle=shuffle
    )
    return train_dataloader


def get_x_physics(x, n=30):
    lower_bound = torch.min(x)
    upper_bound = torch.max(x)
    x_physics = (
        torch.linspace(lower_bound, upper_bound, n).view(-1, 1).requires_grad_(True)
    )
    return x_physics


def train(
    x,
    y,
    model,
    optimizer,
    epochs,
    data_loss_fn=nn.MSELoss(),
    physics_loss_fn=None,
    physics_loss_coef=None,
    save_plot=True,
    draw_loss=True,
):
    dataloader = get_data_for_training(x, y)
    x_physics = get_x_physics(x)
    files = []
    losses = []
    for i in tqdm(range(1, epochs + 1)):
        for x_data, y_data in dataloader:
            # Setting the gradient attribute of each weight to zero
            optimizer.zero_grad()

            # train the data on the small sample of point
            yh = model(x_data)

            # define a loss function (here: mean squared error)
            loss = data_loss_fn(yh, y_data)

            if physics_loss_fn is not None:
                y_physics = model(x_physics)
                physics_loss = physics_loss_fn(x_physics, y_physics)
                loss += physics_loss_coef * physics_loss

            losses.append(loss.item())
            # Computing the gradient
            loss.backward()

            # Adjusting the weights using Adam
            optimizer.step()

            # save a plot as training progresses
        if save_plot and (i % 10 == 0):
            # get a prediciton on the full domain
            yh = model(x).detach()

            # plot and save the prediction
            plot_result(x=x, y=y, x_data=x_data, y_data=y_data, yh=yh, i=i)
            fname = "./img/nn_%d.png" % i
            plt.savefig(fname, bbox_inches="tight", facecolor="white")
            files.append(fname)
            plt.close()

    if draw_loss:
        plt.plot(losses)
        plt.show()

    return files
