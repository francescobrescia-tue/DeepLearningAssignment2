# import numpy as np
# import torch
# import random
# from torch_geometric.data import Data
# import glob
# from torch_geometric.loader import DataLoader


# def load_trajectory(filename, task):
#     """
#     This function loads a trajectory from a given file and returns the trajectory and energy data.
#     If the task is 'task_3', it also returns the framework data.

#     Parameters:
#     filename (str): The name of the file from which to load the trajectory.
#     task (str): The task for which the trajectory is being loaded.
#                 This should be one of 'task_1', 'task_2', or 'task_3'.

#     Returns:
#     tuple: Depending on the task, the function returns:
#            - (trajectory, energy) for 'task_1' and 'task_2'
#            - (trajectory, framework, energy) for 'task_3'
#     """
#     traj = np.load(filename)
#     if task == "task_1" or task == "task_2":
#         trajectory = traj["trajectory"]
#         energy = traj["energy"]
#         return trajectory, energy
#     if task == "task_3":
#         trajectory = traj["trajectory"]
#         framework = traj["framework"]
#         energy = traj["energy"]
#         return trajectory, framework, energy


# def minimum_image_distance(pos1, pos2, box_length):
#     """
#     Compute the distance between two points with the minimum image convention.

#     Parameters:
#     pos1, pos2: numpy arrays representing the positions of the two points.
#     box_length: float representing the length of one side of the box.

#     Returns:
#     float representing the distance between the two points.
#     """
#     delta = pos2 - pos1
#     delta = delta - box_length * np.round(delta / box_length)
#     return np.sqrt(np.sum(delta**2))


# class NBodyDataset:
#     """
#     NBodyDataset

#     """

#     # def __init__(
#     #     self, partition="train", max_samples=1e8, dataset_name="se3_transformer"
#     # ):
#     #     self.partition = partition
#     #     if self.partition == "val":
#     #         self.sufix = "valid"
#     #     else:
#     #         self.sufix = self.partition
#     #     self.dataset_name = dataset_name
#     #     if dataset_name == "nbody":
#     #         self.sufix += "_charged5_initvel1"
#     #     elif dataset_name == "nbody_small" or dataset_name == "nbody_small_out_dist":
#     #         self.sufix += "_charged5_initvel1small"
#     #     else:
#     #         raise Exception("Wrong dataset name %s" % self.dataset_name)

#     #     self.max_samples = int(max_samples)
#     #     self.dataset_name = dataset_name
#     #     self.data, self.edges = self.load()

#     def __init__(self, file_paths, max_len=40, box_length=20.0, cutoff=10.0):

#         self.file_paths = file_paths
#         self.max_len = max_len
#         self.box_length = box_length
#         self.cutoff = cutoff
#         self.task = "task_2"
#         self.data = self.load()

#     # def load(self):
#     #     loc = np.load("n_body_system/dataset/loc_" + self.sufix + ".npy")
#     #     vel = np.load("n_body_system/dataset/vel_" + self.sufix + ".npy")
#     #     edges = np.load("n_body_system/dataset/edges_" + self.sufix + ".npy")
#     #     charges = np.load("n_body_system/dataset/charges_" + self.sufix + ".npy")

#     #     loc, vel, edge_attr, edges, charges = self.preprocess(loc, vel, edges, charges)
#     #     print("loc shape: ", loc.shape)
#     #     print("vel shape: ", vel.shape)
#     #     print("edge_attr shape: ", edge_attr.shape)
#     #     print("edges : ", edges)
#     #     print("charges shape: ", charges.shape)
#     #     return (loc, vel, edge_attr, charges), edges

#     def load(self):

#         data = []

#         for path in self.file_paths:

#             trajectory, energy = load_trajectory(path, self.task)
#             data += self.create_trajectory_data_list(trajectory)
#         return data

#     def create_data_from_particles(self, particles, next_particles):
#         """
#         Particles is a np.ndarray with shape (n_atoms, [pos_x, pos_y, vel_x, vel_y, charge]).
#         Create torch_geometric.data.Data object from particles.
#         """
#         # Create the edge list
#         num_particles = particles.shape[0]
#         edges = []
#         edge_attr = []

#         for i in range(num_particles):
#             for j in range(num_particles):
#                 if i != j:  # Avoid self-loops

#                     # Append edge if distance is below cutoff
#                     pos1 = particles[i, :2]
#                     pos2 = particles[j, :2]
#                     charge1 = particles[i, 4]
#                     charge2 = particles[j, 4]
#                     distance = minimum_image_distance(pos1, pos2, self.box_length)

#                     edges.append([i, j])
#                     if (
#                         distance < self.cutoff
#                     ):  # TODO Per essere corretto non bisognerebbe creare l'edge se la distanza è maggiore del cutoff, ma facendo cosi ci sono errori in alcuni casi limite (annuncio canvas). Ho messo quindi rami con pesi negativi (distanze) per gli edge che non dovrebbero esistere. In questo modo il modello dovrebbe imparare a non considerarli.
#                         edge_attr.append([distance, (charge1 * charge2).item()])
#                     else:
#                         edge_attr.append(
#                             [-1.0, 0]
#                         )  # Use -1.0 as padding value for distance and 0 for charge product

#         loc = particles[:, :2]
#         loc = torch.tensor(loc, dtype=torch.float)

#         vel = particles[:, 2:4]
#         vel = torch.tensor(vel, dtype=torch.float)

#         # norm of the velocity vector as node feature
#         vel_norm = np.linalg.norm(vel, axis=1)
#         x = torch.tensor(vel_norm, dtype=torch.float).unsqueeze(1)

#         # edge_attr = np.array(edge_attr)[:, np.newaxis]
#         edge_attr = torch.tensor(edge_attr, dtype=torch.float)

#         charge = particles[:, 4][:, np.newaxis]
#         charge = torch.tensor(charge, dtype=torch.float)

#         edges = torch.tensor(edges).t().contiguous()
#         # edges = list(zip(*edges))  # Transpose the list of edges
#         # edges = [list(edges[0]), list(edges[1])]
#         # print(f"Edges: {edges}")

#         next_loc = next_particles[:, :2]
#         next_loc = torch.tensor(next_loc, dtype=torch.float)

#         # Create Data object TODO add node features
#         data = Data(
#             loc=loc,
#             vel=vel,
#             x=x,
#             edges=edges,
#             edge_attr=edge_attr,
#             # charge=charge,
#             next_loc=next_loc,
#         )

#         # Validate data object
#         data.validate(raise_on_error=True)

#         return data

#     def create_trajectory_data_list(self, trajectory):
#         """
#         Create a list of data from a trajectory.

#         Parameters:
#         trajectory: numpy array representing the trajectory.

#         Returns:
#         list: A list of data from the trajectory.
#         """
#         data_list = []
#         for i in range(self.max_len - 1):
#             data = self.create_data_from_particles(trajectory[i], trajectory[i + 1])
#             data_list.append(data)

#         return data_list

#     # def preprocess(self, loc, vel, edges, charges):
#     #     # cast to torch and swap n_nodes <--> n_features dimensions
#     #     loc, vel = torch.Tensor(loc).transpose(2, 3), torch.Tensor(vel).transpose(2, 3)
#     #     n_nodes = loc.size(2)
#     #     loc = loc[0 : self.max_samples, :, :, :]  # limit number of samples
#     #     vel = vel[0 : self.max_samples, :, :, :]  # speed when starting the trajectory
#     #     charges = charges[0 : self.max_samples]
#     #     edge_attr = []

#     #     # Initialize edges and edge_attributes
#     #     rows, cols = [], []
#     #     for i in range(n_nodes):
#     #         for j in range(n_nodes):
#     #             if i != j:
#     #                 edge_attr.append(edges[:, i, j])
#     #                 rows.append(i)
#     #                 cols.append(j)
#     #     edges = [rows, cols]
#     #     edge_attr = (
#     #         torch.Tensor(edge_attr).transpose(0, 1).unsqueeze(2)
#     #     )  # swap n_nodes <--> batch_size and add nf dimension

#     #     return (
#     #         torch.Tensor(loc),
#     #         torch.Tensor(vel),
#     #         torch.Tensor(edge_attr),
#     #         edges,
#     #         torch.Tensor(charges),
#     #     )

#     # def set_max_samples(self, max_samples):
#     #     # self.max_samples = int(max_samples)
#     #     self.data, self.edges = self.load()

#     # def get_n_nodes(self):
#     #     return self.data[0].size(1)

#     def __getitem__(self, i):

#         data = self.data[i]

#         """
#         data = Data(
#             loc=loc,
#             vel=vel,
#             x=x,
#             edges=edges,
#             edge_attr=edge_attr,
#             # charge=charge,
#             next_loc=next_loc,
#         )
#         """

#         # return (
#         #     data.loc,
#         #     data.vel,
#         #     data.x,
#         #     data.edges,
#         #     data.edge_attr,
#         #     # data.charge,
#         #     data.next_loc,
#         # )
#         return data

#     def __len__(self):
#         return len(self.data)

#     # def get_edges(self, batch_size, n_nodes):
#     #     print(self.edges)
#     #     edges = [torch.LongTensor(self.edges[0]), torch.LongTensor(self.edges[1])]
#     #     print(edges)
#     #     if batch_size == 1:
#     #         return edges
#     #     elif batch_size > 1:
#     #         rows, cols = [], []
#     #         for i in range(batch_size):
#     #             rows.append(edges[0] + n_nodes * i)
#     #             cols.append(edges[1] + n_nodes * i)
#     #         edges = [torch.cat(rows), torch.cat(cols)]
#     #     return edges


# if __name__ == "__main__":
#     train_paths = glob.glob("data/task1_2/train/*.npz")
#     max_len = 40
#     box_length = 20.0
#     cutoff = 10.0

#     paths = glob.glob("data/task1_2/train/*.npz")

#     train_paths = paths[: int(len(paths) * 0.8)]
#     val_paths = paths[int(len(paths) * 0.8) :]

#     dataset_train = NBodyDataset(train_paths)
#     loader_train = DataLoader(
#         dataset_train, batch_size=100, shuffle=False, drop_last=True
#     )  # TODO: shuffle=True

#     dataset_val = NBodyDataset(val_paths)
#     loader_val = DataLoader(dataset_val, batch_size=100, shuffle=False)

#     dataset_test = NBodyDataset(val_paths)
#     loader_test = DataLoader(dataset_test, batch_size=100, shuffle=False)

#     # dataset_train = NBodyDataset(
#     #     train_paths, max_len, box_length, cutoff
#     # )  # .get_edges(1, 2

#     # dataset_loader = DataLoader(dataset_train, batch_size=1, shuffle=True)

#     # loc, vel, x, edges, edge_attr, next_loc = dataset_train[0]
#     data = dataset_train[0]
#     print(data)

#     print(f"dataset size: {len(dataset_train)}")

#     # print(f"loc: {loc.shape}")
#     # print(f"vel: {vel.shape}")
#     # print(f"x: {x.shape}")
#     # print(f"edges: {edges.shape}")
#     # print(f"edge_attr: {edge_attr.shape}")
#     # print(f"next_loc: {next_loc.shape}")

#     for data in loader_train:
#         print(data)
#         break

#     # print(
#     #     dataset_train[0][0].shape,
#     #     dataset_train[0][1].shape,
#     #     dataset_train[0][2].shape,
#     #     dataset_train[0][3].shape,
#     #     dataset_train[0][4].shape,
#     # )
#     # edges = dataset_train.get_edges(1, 5)


# __________________________________________

import numpy as np
import torch
from torch_geometric.data import Data
import glob
from torch_geometric.loader import DataLoader


def load_trajectory(filename, task):
    traj = np.load(filename)
    if task == "task_1" or task == "task_2":
        trajectory = traj["trajectory"]
        energy = traj["energy"]
        return trajectory, energy
    if task == "task_3":
        trajectory = traj["trajectory"]
        framework = traj["framework"]
        energy = traj["energy"]
        return trajectory, framework, energy


def minimum_image_distance(pos1, pos2, box_length):
    delta = pos2 - pos1
    delta = delta - box_length * np.round(delta / box_length)
    return np.sqrt(np.sum(delta**2))


class NBodyDataset:
    def __init__(self, file_paths, max_len=40, box_length=20.0, cutoff=10.0):
        self.file_paths = file_paths
        self.max_len = max_len
        self.box_length = box_length
        self.cutoff = cutoff
        self.task = "task_2"
        self.data = self.load()

    def load(self):
        data = []
        for path in self.file_paths:
            trajectory, energy = load_trajectory(path, self.task)
            data += self.create_trajectory_data_list(trajectory)
        return data

    def create_data_from_particles(self, particles, next_particles):
        num_particles = particles.shape[0]
        edges = []
        edge_attr = []

        for i in range(num_particles):
            for j in range(num_particles):
                if i != j:
                    pos1 = particles[i, :2]
                    pos2 = particles[j, :2]
                    charge1 = particles[i, 4]
                    charge2 = particles[j, 4]
                    distance = minimum_image_distance(pos1, pos2, self.box_length)
                    edges.append([i, j])
                    if distance < self.cutoff:
                        edge_attr.append([distance, (charge1 * charge2).item()])
                    else:
                        edge_attr.append([-1.0, 0])

        loc = torch.tensor(particles[:, :2], dtype=torch.float)
        vel = torch.tensor(particles[:, 2:4], dtype=torch.float)
        vel_norm = torch.tensor(
            np.linalg.norm(vel, axis=1), dtype=torch.float
        ).unsqueeze(1)
        x = vel_norm
        edge_attr = torch.tensor(edge_attr, dtype=torch.float)
        edges = torch.tensor(edges).t().contiguous()
        next_loc = torch.tensor(next_particles[:, :2], dtype=torch.float)
        next_vel = torch.tensor(next_particles[:, 2:4], dtype=torch.float)

        data = Data(
            loc=loc,
            vel=vel,
            x=x,
            edge_index=edges,
            edge_attr=edge_attr,
            next_loc=next_loc,
            next_vel=next_vel,
        )
        return data

    def create_trajectory_data_list(self, trajectory):
        data_list = []
        for i in range(self.max_len - 1):
            data = self.create_data_from_particles(trajectory[i], trajectory[i + 1])
            data_list.append(data)
        return data_list

    def __getitem__(self, i):
        return self.data[i]

    def __len__(self):
        return len(self.data)


if __name__ == "__main__":
    train_paths = glob.glob("data/task1_2/train/*.npz")
    max_len = 40
    box_length = 20.0
    cutoff = 10.0

    paths = glob.glob("data/task1_2/train/*.npz")

    train_paths = paths[: int(len(paths) * 0.8)]
    val_paths = paths[int(len(paths) * 0.8) :]

    dataset_train = NBodyDataset(train_paths)
    loader_train = DataLoader(
        dataset_train, batch_size=100, shuffle=False, drop_last=True
    )
    dataset_val = NBodyDataset(val_paths)
    loader_val = DataLoader(dataset_val, batch_size=100, shuffle=False)
    dataset_test = NBodyDataset(val_paths)
    loader_test = DataLoader(dataset_test, batch_size=100, shuffle=False)

    data = dataset_train[0]
    print(data)

    print(f"dataset size: {len(dataset_train)}")

    for data in loader_train:
        print(data)
        break
