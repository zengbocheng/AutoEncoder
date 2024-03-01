"""
@author: bochengz
@date: 2024/02/26
@email: bochengzeng@bochengz.top
"""
from common.data_handler import DataHandler
from torch.utils.data import DataLoader, Dataset
import os.path as osp
import h5py
import torch
import numpy as np


class GridCylinderDataHandler(DataHandler):
    """
    Data handler for cylinder flow system on regular grid
    """

    def __init__(self):
        super().__init__()
        self.mu = None
        self.std = None

    def create_tr_loader(self, file_path, n_data, time_window, batch_size,
                         num_workers):
        """
        Create training/validation data loader for cylinder flow system.
        For a single trajectory of simulation, the total time series is \
        sub-chunked into smaller time slices.

        Args:
            file_path (str): Path of training/validation HDF5 file
            n_data (int): Number of trajectories for data loader
            time_window (int): The size of each time slice
            batch_size (int): Batch size of data loader
            num_workers (int): Number of processes of data loader

        Returns:
            data_loader (DataLoader): Training/validation data loader
        """
        data, visc = self.read_h5file(
            file_path,
            n_data,
            time_window
        )

        self.mu = torch.tensor([
            torch.mean(data[:, :, 0]), torch.mean(data[:, :, 1]),
            torch.mean(data[:, :, 2]), torch.mean(visc)
        ])
        self.std = torch.tensor([
            torch.std(data[:, :, 0]), torch.std(data[:, :, 1]),
            torch.std(data[:, :, 2]), torch.std(visc)
        ])

        dataset = self.GridCylinderDataset(data, visc)
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                                 num_workers=num_workers)
        return data_loader

    def create_val_loader(self, file_path, n_data, time_window, batch_size,
                          num_workers):
        data, visc = self.read_h5file(
            file_path,
            n_data,
            time_window
        )

        dataset = self.GridCylinderDataset(data, visc)
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                                 num_workers=num_workers)
        return data_loader

    def create_te_loader(self, file_path, n_data, time_window):
        data, visc = self.read_h5file(
            file_path,
            n_data,
            time_window
        )

        dataset = self.GridCylinderDataset(data, visc)
        data_loader = DataLoader(dataset, batch_size=1, shuffle=False)
        return data_loader

    def read_h5file(self, file_path, n_data, time_window):
        if not osp.isfile(file_path):
            raise FileNotFoundError('{} - {} not found!'
                                    .format(self.read_h5file.__name__,
                                            file_path))

        samples = []
        viscosities = []
        n_count = 0
        with h5py.File(file_path, 'r') as f:
            for key in f.keys():
                viscosity = (2.0 / float(key))
                ux = torch.tensor(np.array(f[key + '/ux']))  # (T, h, w)
                uy = torch.tensor(np.array(f[key + '/uy']))
                p = torch.tensor(np.array(f[key + '/p']))
                traj = torch.stack([ux, uy, p], dim=1)  # (T, 3, h, w)
                for i in range(0, traj.shape[0] - time_window + 1):
                    samples.append(traj[i: i + time_window])
                    viscosities.append(viscosity)

                n_count += 1
                if n_count >= n_data:
                    break

        if time_window > 1:
            data = torch.stack(samples, dim=0)  # (n, tw, 3, h, w)
        else:
            data = torch.cat(samples, dim=0)  # (n, 3, h, w)
        visc = torch.tensor(viscosities).unsqueeze(-1)
        return data, visc

    class GridCylinderDataset(Dataset):
        """
        Args:
            data (torch.Tensor): shape (n, tw, 3, h, w) or (n, 3, h, w)
            visc (torch.Tensor): shape (n, 1)
        """

        def __init__(self, data, visc):
            self.data = data
            self.visc = visc

        def __len__(self):
            return self.data.shape[0]

        def __getitem__(self, i):
            return self.data[i], self.visc[i]
