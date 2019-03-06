import copy
import logging
import os

import torch

from utils import start_idx_from_lengths


class Trajectories:
    """
    Class used to represent trajectories (sequence of distribution/observation on a graph).

    A bit of boilerplate code is needed to mask some trajectories (train/test) without duplicating the underlying tensors.
    Trajectory distributions are stored in a sparse format (weights, indices) and converted to dense when accessed trajectories[i]
    Optional `traversed_edges` can contain the edges traversed between each jump of the trajectory
    """

    def __init__(
        self,
        weights: torch.Tensor,
        indices: torch.Tensor,
        num_nodes: int,
        lengths: torch.Tensor,
        traversed_edges: torch.Tensor = None,
        pairwise_node_distances: torch.Tensor = None,
    ):
        """Create a new trajectories object, can be masked to have access to only a subset of the dataset

        :param weights:
            each row should sum to 1
            shape: num_observations x k
        :param weights: indices of the nodes
            shape: num_observations x k
        :param lengths: contains the number of observations in each trajectory
            shape: num_trajectories
        :param traversed_edges:
            uses -1 token for padding
            shape: (num_observations - num_trajectories) x max_path_length
        Args:
            weights (torch.Tensor): [num_observations, k] weights on the nodes, rows should sum to 1
            indices (torch.Tensor): [num_observations, k] indices of the nodes
            num_nodes (int): number of nodes
            lengths (torch.Tensor): [num_trajectories, ] length of each trajectory
            traversed_edges (torch.Tensor, optional): Defaults to None.
                [num_observations - num_trajectories, max_path_length] Edge ids traversed between two observations (paded with -1)
            pairwise_node_distances (torch.Tensor, optional): Defaults to None. [n_node, n_node] distances between nodes on the graph
        """

        assert weights.shape == indices.shape
        assert weights.dim() == 2

        self._weights = weights
        self._indices = indices
        self._lengths = lengths
        self._traversed_edges = traversed_edges
        self._starts = start_idx_from_lengths(lengths)
        self._mask = None
        self.num_nodes = num_nodes
        self.pairwise_node_distances = pairwise_node_distances

        self.device = weights.device
        self._check_device()

        # precomputed fields
        self._index_mapping = None
        self._num_trajectories = None

        # check that the number of legs equals sum of (trajectory lengths - 1 each)
        if traversed_edges is not None:
            assert traversed_edges.shape[0] == self._lengths.sum() - len(self)

    def __len__(self) -> int:
        """Returns number of trajectories

        Returns:
            int: length
        """
        if self._mask is None:
            self._num_trajectories = len(self._lengths)
        else:
            self._num_trajectories = self._mask.long().sum().item()
        return self._num_trajectories

    def __getitem__(self, item):
        """Distributions of the trajectory at item

        Args:
            item (int): index of the trajectory

        Returns:
            torch.Tensor: [traj_length, n_node]
        """
        item = self._mapped_index(item)
        start = self._starts[item]
        length = self._lengths[item]
        observations = torch.zeros([length, self.num_nodes], device=self.device)
        row = torch.arange(length).unsqueeze(1).repeat(1, self._weights.shape[1])
        observations[row, self._indices[start : start + length]] = self._weights[
            start : start + length
        ]
        return observations

    @property
    def lengths(self):
        """Tensor containing length of each non masked trajectory

        Returns:
            torch.Tensor: [n_traj,] lengths
        """
        if self._mask is None:
            return self._lengths
        else:
            return self._lengths[self._mask]

    @property
    def has_traversed_edges(self):
        return self._traversed_edges is not None

    def _mapped_index(self, index):
        """transform an input index to index in the non masked trajectories underlying"""
        if self._mask is None:
            return index
        else:
            if self._index_mapping is None:
                self._index_mapping = self._mask.nonzero()[:, 0]
            return self._index_mapping[index]

    def _traversed_edges_by_trajectory(self, trajectory_id: int) -> torch.Tensor:
        item = self._mapped_index(trajectory_id)
        start = self._starts[item] - item  # -1 leg per trajectory
        length = self._lengths[item] - 1
        traversed_edges = self._traversed_edges[start : start + length]
        return traversed_edges

    def traversed_edges(self, trajectory_id, jump=None):
        traversed_edges = self._traversed_edges_by_trajectory(trajectory_id)
        if jump is not None:
            traversed_edges = traversed_edges[jump]

        traversed_edges = traversed_edges.flatten()
        traversed_edges = traversed_edges[traversed_edges != -1]
        return traversed_edges

    def leg_lengths(self, trajectory_id):
        traversed_edges = self._traversed_edges_by_trajectory(trajectory_id)
        lengths = (traversed_edges != -1).sum(dim=1)
        return lengths

    def leg_shortest_lengths(self, trajectory_id):
        observations = self[trajectory_id]
        num_jumps = self.lengths[trajectory_id] - 1
        min_distances = torch.zeros(num_jumps, device=self.device, dtype=torch.long)

        for jump in range(num_jumps):
            fr_nodes = observations[jump].nonzero().squeeze()
            to_nodes = observations[jump + 1].nonzero().squeeze()
            all_distances = self.pairwise_node_distances[fr_nodes, :][:, to_nodes]
            min_distances[jump] = all_distances[all_distances >= 0].min()

        return min_distances

    def clone(self) -> "Trajectories":
        """Shallow copy"""
        return copy.copy(self)

    def to(self, device: torch.device) -> "Trajectories":
        """Move this Trajectories instance to the required device

        Returns:
            Trajectories: Moved Trajectories
        """
        if self.device == device:
            return self

        moved_trajectories = self.clone()
        moved_trajectories.device = device
        for attribute, value in moved_trajectories.__dict__.items():
            if type(value) is torch.Tensor:
                moved_trajectories.__dict__[attribute] = value.to(device)
        return moved_trajectories

    def _check_device(self):
        """Check that all attributes of the trajectories are on `self.device`

        Raises:
            ValueError: if a tensor is not on the right device
        """
        for attribute, value in self.__dict__.items():
            if hasattr(value, "device") and value.device != self.device:
                raise ValueError(
                    f"Trajectories attribute '{attribute}' is on device '{value.device}' instead of '{self.device}'"
                )

    def with_mask(self, mask):
        """Create a new view of trajectories with only the visible trajectories with True value in the mask tensor

        Args:
            mask (torch.Tensor): mask

        Returns:
            Trajectories:
        """

        if mask is not None and mask.device != self.device:
            mask = mask.to(self.device)

        masked_trajectories = self.clone()
        masked_trajectories._mask = mask
        masked_trajectories._reset_precomputation()
        return masked_trajectories

    def _reset_precomputation(self):
        """Reset to None precomputed values to access properties"""
        self._num_trajectories = None
        self._index_mapping = None

    """ ---------- READ/WRITE FILE ---------- """

    def write_to_directory(self, directory):
        """Save this object in the directory (ignore the mask)
        Writes `lengths.txt` `observations.txt` and optionally `paths.txt`
        see `read_from_files` documentation for the format
        """
        if self._mask is not None:
            logging.warning("Trajectories mask ignored when writing to directory")

        os.makedirs(directory, exist_ok=True)

        with open(os.path.join(directory, "lengths.txt"), "w") as f:
            for i, l in enumerate(self._lengths):
                f.write("{}\t{}\n".format(i, l.item()))

        with open(os.path.join(directory, "observations.txt"), "w") as f:
            f.write("{}\t{}\n".format(*self._indices.shape))
            for row in range(self._indices.shape[0]):
                row_elems = []
                for col in range(self._indices.shape[1]):
                    row_elems.append(self._indices[row, col])
                    row_elems.append(self._weights[row, col])
                f.write("\t".join(map(lambda x: str(x.item()), row_elems)) + "\n")

        with open(os.path.join(directory, "paths.txt"), "w") as f:
            f.write("{}\t{}\n".format(*self._traversed_edges.shape))
            for leg in self._traversed_edges:
                line = "\t".join(str(p.item()) for p in leg if p != -1) + "\n"
                f.write(line)

    @classmethod
    def read_from_files(
        cls, lengths_filename, observations_filename, paths_filename, num_nodes
    ):
        """
        Read trajectories from files `lengths.txt` `observations.txt` and `paths.txt`

        Length file has per line trajectory id and length Example
        ```
        0	9
        1	9
        2	7
        3	8
        4	7
        5	7
        ```

        Observations file start with num_observations, k (point per observation)
        then per line node_id, weight x k. Example:
        ```
        2518	5
        17025	0.22376753215971462	17026	0.2186635904321353	1137	0.18742442008753432	6888	0.20024607632540276	4585	0.16989838099521318
        6888	0.20106576291692577	1137	0.20348475328200213	4585	0.20255400616332436	1139	0.1985437138699239	6887	0.1943517637678238
        14928	0.18319982750248237	1302	0.18136407620166017	14929	0.1979849150163569	628	0.18905104643181994	1303	0.24840013484768056
        ```

        Paths file start with number of paths and maximum length
        Then per line, sequence of traversed edge ids. Example:
        ```
        2254	41
        20343	30411	30413	12311	1946
        1946	8179	30415	24401	24403	1957	8739	1960	24398	24400	20824	20822	20814	19664	19326	19327	26592	19346	29732	26594	13778	20817	13785	26595	26597
        ```
        """

        # read trajectories lengths
        with open(lengths_filename) as f:
            lengths = [int(line.split("\t")[1]) for line in f.readlines()]
            lengths = torch.tensor(lengths)

        # read observations, assume fixed number of observations
        obs_weights, obs_indices = None, None
        with open(observations_filename) as f:
            num_observations, k = map(int, f.readline().split("\t"))
            obs_weights = torch.zeros(num_observations, k)
            obs_indices = torch.zeros(num_observations, k, dtype=torch.long)

            for i, line in enumerate(f.readlines()):
                elements = line.split("\t")
                for n in range(k):
                    obs_indices[i, n] = int(elements[2 * n])
                    obs_weights[i, n] = float(elements[2 * n + 1])

        # read underlying paths
        paths = None
        if paths_filename is not None and os.path.exists(paths_filename):
            with open(paths_filename) as f:
                num_paths, max_path_length = map(int, f.readline().split("\t"))
                paths = torch.zeros([num_paths, max_path_length], dtype=torch.long) - 1
                for i, line in enumerate(f.readlines()):
                    ids = list(map(int, line.split("\t")))
                    if len(ids) == 0:
                        print(i)
                    paths[i, : len(ids)] = torch.tensor(ids)

        return Trajectories(
            weights=obs_weights,
            indices=obs_indices,
            num_nodes=num_nodes,
            lengths=lengths,
            traversed_edges=paths,
        )

