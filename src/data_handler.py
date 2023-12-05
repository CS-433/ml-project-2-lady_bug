import torch
from torch import nn
from torch.utils.data import DataLoader

from tqdm import tqdm


class DataHandler:
    def __init__(
        self,
        x,
        y,
        data_end=None,
        data_step=None,
        physics_n=None,
        batch_size=None,
        shuffle=False,
    ):
        self.x, self.y = x, y
        self.x_data, self.y_data = self.__get_data_for_training(data_end, data_step)

        self.x_test, self.y_test = self.__get_data_for_testing(data_end, data_step)

        print(self.x_data.shape, self.x_test.shape)

        x_physics = self.__get_x_physics(physics_n)
        if batch_size is None:
            batch_size = len(x_physics)
        self.physics_dataloader = DataLoader(
            x_physics, batch_size=batch_size, shuffle=shuffle
        )

    def __get_data_for_training(self, end=None, step=None):
        if end is None:
            end = round(0.2 * len(self.x))
        if step is None:
            step = max(end // 10, 1)
        x_data = self.x[0:end:step]
        y_data = self.y[0:end:step]
        return x_data, y_data
    
    def __get_data_for_testing(self, end=None, step=None):
        if end is None:
            end = round(0.2 * len(self.x))
        if step is None:
            step = max(end // 10, 1)
        
        idx = torch.ones_like(self.x)
        idx[0:end:step] = 0

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
    

class DataHandlerForAllSimulations():
    def __init__(
        self,
        x,
        y,
        data_end=None,
        data_step=None,
        physics_step=50,
        n_simulations=100,
        train_fraction=0.9,
        batch_size=1,
        shuffle=False,
    ):
        self.x, self.y = x[:n_simulations], y[:n_simulations]
        self.n_train_simulations = int(train_fraction * self.x.shape[0])
        
        x_train, y_train = self.__get_data_for_training(data_end, data_step)
        x_physics = self.__get_x_physics(physics_step)
        x_test, y_test = self.__get_data_for_testing()
        
        self.batch_size = batch_size

        self.train_dataloader = DataLoader(
            list(zip(x_train, y_train, x_physics)), batch_size=self.batch_size, shuffle=shuffle
        )
        self.test_dataloader = DataLoader(
            list(zip(x_test, y_test)), batch_size=self.batch_size, shuffle=shuffle
        )

    def __get_data_for_training(self, end=None, step=None):
        if end is None:
            end = round(0.4 * self.x.shape[1])
        if step is None:
            step = max(end // 10, 1)
        x_data, y_data = self.x[:self.n_train_simulations], self.y[:self.n_train_simulations]
        x_data = self.x[:, 0:end:step]
        y_data = self.y[:, 0:end:step]
        return x_data, y_data
    
    def __get_data_for_testing(self):
        x_data, y_data = self.x[self.n_train_simulations:], self.y[self.n_train_simulations:]
        return x_data, y_data

    def __get_x_physics(self, step):
        x_data, y_data = self.x[:self.n_train_simulations], self.y[:self.n_train_simulations]
        x_data = self.x[:, 0:self.n_train_simulations:step].requires_grad_(True)
        return x_data


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
