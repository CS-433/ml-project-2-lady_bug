import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torcheval.metrics.functional import r2_score

from torch.utils.tensorboard import SummaryWriter

from tqdm import tqdm

import matplotlib.pyplot as plt

from src.physics_loss import physics_loss_varied_gamma_n
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

    def loss(self, yh, y_data, x_physics, y_physics, on_epoch_end=False, iter=None):
        loss = 0.0
        if self.data_loss is not None:
            data_loss = self.data_loss(yh, y_data)
            if on_epoch_end:
                self.track_progress(iter, data_loss, "data_loss")
            loss += data_loss
        
        if self.physics_loss is not None:
            physics_loss = self.physics_loss( x_physics, y_physics)
            if on_epoch_end:
                self.track_progress(iter, physics_loss, "physics_loss")
            loss += physics_loss
        
        if on_epoch_end:
            self.track_progress(iter, loss, "total_loss")
        return loss

    def track_progress(self, iter, variable, variable_name):
        self.writer.add_scalar(variable_name, variable, iter)

    def train(self, epochs):
        for i in tqdm(range(epochs)):
            for batch_idx, x_physics in enumerate(self.data_handler.physics_dataloader):
                x_data = self.data_handler.x_data
                y_data = self.data_handler.y_data

                self.optimizer.zero_grad()

                yh = self.model(x_data)
                y_physics = self.model(x_physics)

                if batch_idx == len(self.data_handler.physics_dataloader) - 1:
                    # Tensorboard callbacks for the parts of loss at the end of each epoch
                    loss = self.loss(yh, y_data, x_physics, y_physics, on_epoch_end=True, iter=i)
                else:
                    loss = self.loss(yh, y_data, x_physics, y_physics)
                loss.backward()

                self.optimizer.step()
                if self.scheduler is not None:
                    self.scheduler.step(loss.item())

            # Tensorboard callback for learning rate
            self.track_progress(i, float(self.optimizer.param_groups[0]["lr"]), "learning_rate")
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

    def score(self):
        yh = self.model(self.data_handler.x_test)
        return r2_score(yh, self.data_handler.y_test)


class RunAllSimulations(Run):
    def __init__(self, log_dir):
        super().__init__(log_dir)

    def train(self, epochs):
        print("Total num of the batches: ", len(self.data_handler.train_dataloader))
        for i in tqdm(range(epochs)):
            for batch_idx, batch in enumerate(self.data_handler.train_dataloader):
                """
                batch is a single simulation -> batch_size = 1, x has shape (batch_size, n_timesteps, feature_dim)
                where features are (timestep, n, Gamma)
                """
                x_train, y_train, x_physics = batch[0][0], batch[1][0], batch[2][0]
                y_train = y_train.view(-1, 1)
                self.optimizer.zero_grad()

                yh = self.model(x_train)
                y_physics = self.model(x_physics)

                if batch_idx == len(self.data_handler.train_dataloader) - 1:
                    # Tensorboard callbacks for the parts of loss at the end of each epoch
                    loss = self.loss(yh, y_train, x_physics, y_physics, on_epoch_end=True, iter=i)
                else:
                    loss = self.loss(yh, y_train, x_physics, y_physics)
                loss.backward()

                self.optimizer.step()
                if self.scheduler is not None:
                    self.scheduler.step(loss.item())

            # Tensorboard callback for learning rate
            self.track_progress(i, float(self.optimizer.param_groups[0]["lr"]), "learning_rate")

    def test(self, return_all_runs=False):
        """
        For testing we predict approximation for all test simulations and evaluate the quality in terms of r2 score.
        The simulation with the highest r2 score between prediction and target function is considered "best", 
        simulation with the lowest r2 score - "worst"
        """
        r2_per_simulation = []
        yh_array = []
        y_test_array = []
        for x_test, y_test in self.data_handler.test_dataloader:
            yh = self.model(x_test[0])
            r2_per_simulation.append(r2_score(yh, y_test.T))

            yh_array.append(yh)
            y_test_array.append(y_test.T)
        
        
        r2_per_simulation = np.array(r2_per_simulation)
        max_idx = np.argmax(r2_per_simulation)
        min_idx = np.argmin(r2_per_simulation)
        print("R2 best simulation: ", np.max(r2_per_simulation))
        print("R2 worst simulation: ", np.min(r2_per_simulation))
        # we use return_all_runs if we want to visualize all predictions, otherwise only the best and worst ones are plotted
        if return_all_runs:
            return yh_array, y_test_array
        return r2_per_simulation, yh_array[max_idx], y_test_array[max_idx], yh_array[min_idx], y_test_array[min_idx]
