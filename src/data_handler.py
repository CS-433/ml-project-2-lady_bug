import torch
from torch import nn
from torch.utils.data import DataLoader, Sampler, TensorDataset
from tqdm import tqdm

from src.constants import OOD_THRESHOLD


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


class WeightedSampler(Sampler):
    """Custor Sampler for more frequent sampling of out-of-distribution simulations"""
    def __init__(self, dataset, ood_ids, odd_weight=2., normal_weight=1.):
        """
        ood_ids is a set of indices of out-of-distubution simulations
        """
        self.indices = list(range(len(dataset)))
        self.num_samples = len(dataset)
        weights = [odd_weight if i in ood_ids else normal_weight for i in self.indices] # weights are proportional to the frequency of ood runs
        self.weights = torch.tensor(weights, dtype=torch.double)
        
    def __iter__(self):
        count = 0
        # re-sample indices of simulations to balance normal and ood simulations
        index = [self.indices[i] for i in torch.multinomial(self.weights, self.num_samples, replacement=True)]
        while count < self.num_samples:
            yield index[count]
            count += 1
    
    def __len__(self):
        return self.num_samples


class DataHandlerForAllSimulations():
    def __init__(
        self,
        x,
        y,
        data_end=None,
        data_step=None,
        physics_step=50,
        resample_ood_runs=False,
        train_fraction=0.9,
        batch_size=1,
        shuffle=False,
        ood_weight=2.,
        normal_weight=1.
    ):
        self.x, self.y = x, y
        self.n_train_simulations = int(train_fraction * x.shape[0])
        
        x_train, y_train = self.__get_data_for_training(data_end, data_step)
        x_physics = self.__get_x_physics(physics_step)
        x_test, y_test = self.x[self.n_train_simulations:], self.y[self.n_train_simulations:]

        self.batch_size = batch_size

        train_dataset = TensorDataset(x_train, y_train, x_physics)

        if ood_weight == 0 or normal_weight == 0: 
            odd_test_ids = self.__get_ood_ids_test_data(data_end, data_step)
            include_ids = torch.ones(y_test.shape[0]) * normal_weight
            include_ids[odd_test_ids] = ood_weight
            include_ids = include_ids.bool()
            x_test, y_test = x_test[include_ids], y_test[include_ids]

        test_dataset = TensorDataset(x_test, y_test)

        if resample_ood_runs:
            ood_ids = self.__get_ood_ids(y_train)
            ood_sampler = WeightedSampler(train_dataset, ood_ids, ood_weight, normal_weight)
            self.train_dataloader = DataLoader(
                train_dataset, batch_size=self.batch_size, sampler=ood_sampler, shuffle=shuffle
            )
        else:
            self.train_dataloader = DataLoader(
                train_dataset, batch_size=self.batch_size, shuffle=shuffle
            )
        self.test_dataloader = DataLoader(
            test_dataset, batch_size=self.batch_size, shuffle=False
        )
    
    def __get_ood_ids(self, y_train):
        ood_ids = set()
        for i in range(y_train.shape[0]):
            if y_train[i, -1] <= OOD_THRESHOLD:
                ood_ids.add(i)
        return ood_ids

    def __get_data_for_training(self, end=None, step=None, resample_ood_runs=False):
        if end is None:
            end = round(0.4 * self.x.shape[1])
        if step is None:
            step = max(end // 20, 1)
        
        x_data, y_data = self.x[:self.n_train_simulations], self.y[:self.n_train_simulations]
        x_data = x_data[:, 0:end:step]
        y_data = y_data[:, 0:end:step]
        return x_data, y_data
    
    def __get_ood_ids_test_data(self, end, step):
        if end is None:
            end = round(0.4 * self.x.shape[1])
        if step is None:
            step = max(end // 20, 1)
        
        y_data = self.y[self.n_train_simulations:]
        y_data = y_data[:, 0:end:step]
        return list(self.__get_ood_ids(y_data))

    def __get_x_physics(self, step, resample_ood_runs=False):
        x_data, y_data = self.x[:self.n_train_simulations], self.y[:self.n_train_simulations]
        x_data = x_data[:, 0:self.n_train_simulations:step].requires_grad_(True)
        return x_data


class RandomPointsIterator:
    def __init__(self, lower_bound, upper_bound, n):
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.n = n

        self.start_idx, self.last_idx = 0, 1

    def __iter__(self):
        return self
    
    def __len__(self):
        return self.last_idx
    
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
        self, x, y, data_start=0, data_end=None, data_step=None, physics_n=None, batch_size=None, shuffle=False
    ):
        super().__init__(x, y, data_start, data_end, data_step)
        self.physics_dataloader = self.__random_points_iterator()

    def __random_points_iterator(self, n=None):
        if n is None:
            n = round(0.4 * len(self.x))

        lower_bound = torch.min(self.x).item()
        upper_bound = torch.max(self.x).item()
        return RandomPointsIterator(lower_bound, upper_bound, n)
