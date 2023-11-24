import torch
from torch import nn
from torch.utils.data import DataLoader

from torch.utils.tensorboard import SummaryWriter

from tqdm import tqdm

import matplotlib.pyplot as plt
from src.utils import plot_result


class Run:
    def __init__(self, log_dir):
        self.data_handler = None

        self.model = None
        self.optimizer = None
        self.scheduler = None

        self.data_loss = None
        self.physics_loss = None

        self.writer = SummaryWriter("runs/" + log_dir)

    def loss(self, yh, y_data, x_physics, y_physics):
        if self.physics_loss is None:
            return self.data_loss(yh, y_data)
        
        return self.data_loss(yh, y_data) + self.physics_loss(x_physics, y_physics)

    def __track_progress(self, iter, loss):
        self.writer.add_scalar("training loss", loss, iter)

    def __step(self):
        for x_physics in self.data_handler.physics_dataloader:
            x_data = self.data_handler.x_data
            y_data = self.data_handler.y_data

            self.optimizer.zero_grad()

            yh = self.model(x_data)
            y_physics = self.model(x_physics)

            loss = self.loss(yh, y_data, x_physics, y_physics)
            loss.backward()

            self.optimizer.step()
            if self.scheduler is not None:
                self.scheduler.step(loss.item())
        return loss.item()

    def train(self, epochs):
        for i in tqdm(range(epochs)):
            loss = self.__step()
            self.__track_progress(i, loss)
        self.__save_img()

    def __save_img(self):
        # get a prediciton on the full domain
        yh = self.model(self.data_handler.x).detach()
        # plot and save the prediction
        self.writer.add_figure(
            "predicted function",
            plot_result(
                x=self.data_handler.x,
                y=self.data_handler.y,
                x_data=self.data_handler.x_data,
                y_data=self.data_handler.y_data,
                yh=yh,
            ),
        )
