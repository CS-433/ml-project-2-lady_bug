import torch
from torch import nn
from torch.utils.data import DataLoader

from tqdm import tqdm

import matplotlib.pyplot as plt
from src.utils import plot_result


class DataHandler:
    def __init__(self, x, y):
        self.x, self.y = x, y
        self.x_data, self.y_data = self.__get_data_for_training()
        self.dataloader =DataLoader(list(zip(self.x_data, self.y_data)), batch_size=1000, shuffle=True)
        self.x_physics = self.__get_x_physics()

    def __get_data_for_training(self, end=40, step=4):
        x_data = self.x[0:end:step]
        y_data = self.y[0:end:step]
        return x_data, y_data

    def __get_x_physics(self, n=30):
        lower_bound = torch.min(self.x)
        upper_bound = torch.max(self.x)
        x_physics = (
            torch.linspace(lower_bound, upper_bound, n).view(-1, 1).requires_grad_(True)
        )
        return x_physics


class Run:
    def __init__(self, x, y):
        self.data_handler = DataHandler(x, y)  # probably delete later

        self.model = None
        self.optimizer = None

        self.data_loss = None
        self.physics_loss = None

        self.writer = None

        self.files = []  # delete later

    def loss(self, yh, y_data, x_physics, y_physics):
        if self.physics_loss is None:
            return self.data_loss(yh, y_data)

        return self.data_loss(yh, y_data) + self.physics_loss(x_physics, y_physics)

    def __track_progress(self, iter):
        if iter % 10 == 0:
            # get a prediciton on the full domain
            yh = self.model(self.data_handler.x).detach()

            # plot and save the prediction
            plot_result(
                x=self.data_handler.x,
                y=self.data_handler.y,
                x_data=self.data_handler.x_data,
                y_data=self.data_handler.y_data,
                yh=yh,
                i=iter,
            )
            fname = "./img/nn_%d.png" % iter
            plt.savefig(fname, bbox_inches="tight", facecolor="white")
            self.files.append(fname)
            plt.close()

    def __step(self):
        for x_data, y_data in self.data_handler.dataloader:
            self.optimizer.zero_grad()

            yh = self.model(x_data)
            y_physics = self.model(self.data_handler.x_physics)

            loss = self.loss(yh, y_data, self.data_handler.x_physics, y_physics)
            loss.backward()

            self.optimizer.step()

    def train(self, epochs):
        for i in tqdm(range(1, epochs + 1)):
            self.__step()
            self.__track_progress(i)
