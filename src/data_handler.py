import torch
from torch import nn
from torch.utils.data import DataLoader

from tqdm import tqdm


class DataHandler:
    def __init__(
        self,
        x,
        y,
        data_start=0,
        data_end=None,
        data_step=None,
        physics_n=None,
        batch_size=None,
        shuffle=False,
    ):
        self.x, self.y = x, y
        self.x_data, self.y_data = self.__get_data_for_training(data_start, data_end, data_step)

        self.x_test, self.y_test = self.__get_data_for_testing(data_start, data_end, data_step)

        print(self.x_data.shape, self.x_test.shape)

        x_physics = self.__get_x_physics(physics_n)
        if batch_size is None:
            batch_size = len(x_physics)
        self.physics_dataloader = DataLoader(
            x_physics, batch_size=batch_size, shuffle=shuffle
        )

    def __get_data_for_training(self, start=0, end=None, step=None):
        if end is None:
            end = round(0.2 * len(self.x))
        if step is None:
            step = max(end // 10, 1)
        x_data = self.x[start:end:step]
        y_data = self.y[start:end:step]
        return x_data, y_data
    
    def __get_data_for_testing(self, start=0, end=None, step=None):
        if end is None:
            end = round(0.2 * len(self.x))
        if step is None:
            step = max(end // 10, 1)
        
        idx = torch.ones_like(self.x)
        idx[start:end:step] = 0

        x_data = self.x[idx.bool()].view(-1, 1)
        y_data = self.y[idx.bool()].view(-1, 1)
        return x_data, y_data

    def __get_x_physics(self, n=None):
        if n is None:
            n = round(0.4 * len(self.x))

        lower_bound = torch.min(self.x).item()
        upper_bound = torch.max(self.x).item()
        x_physics = (
            torch.linspace(lower_bound, upper_bound, n).view(-1, 1).requires_grad_(True)
        )
        return x_physics


class RandomPointsIterator:
    def __init__(self, lower_bound, upper_bound, n):
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.n = n

        self.start_idx, self.last_idx = 0, 1

    def __iter__(self):
        return self

    def __next__(self):
        if self.start_idx < self.last_idx:
            x_physics = (
                torch.distributions.uniform.Uniform(self.lower_bound, self.upper_bound)
                .sample([self.n])
                .view(-1, 1)
                .requires_grad_(True)
            )
            self.start_idx += 1
            return x_physics
        self.start_idx = 0
        raise StopIteration


class RandomSamplingDataHandler(DataHandler):
    def __init__(
        self, x, y, data_end=None, data_step=None, batch_size=None, shuffle=False
    ):
        super().__init__(x, y, data_end, data_step)
        self.physics_dataloader = self.__random_points_iterator()

    def __random_points_iterator(self, n=None):
        if n is None:
            n = round(0.4 * len(self.x))

        lower_bound = torch.min(self.x).item()
        upper_bound = torch.max(self.x).item()
        return RandomPointsIterator(lower_bound, upper_bound, n)
