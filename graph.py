import copy
import logging
import os

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pygsp
import scipy
import torch
from typing import Union, List
from torch_scatter import scatter_add, scatter_max
import matplotlib as mpl

import plot as gnx_plot
from debug import assert_allclose, any_nan
from utils import numpify, sample


class Graph:
    """Graph represenations

    Represents a graph with edge senders and receivers, features on the nodes and features on the edges.
    Use `update` method to modify this graph.
    """

    def __init__(self,
                 senders,
                 receivers,
                 nodes,
                 edges,
                 n_node=None,
                 n_edge=None):
        self.senders = senders
        self.receivers = receivers
        self.nodes = nodes
        self.edges = edges
        self.n_node = n_node or nodes.shape[0]
        self.n_edge = n_edge or senders.shape[0]

        self.device = senders.device

        self._pygsp = None
        self._coords = None
        self._augmented_edge_features = None
        self._distances = None
        self._non_backtracking_random_walk_graph = None
        self._non_backtracking_edge_senders = None
        self._non_backtracking_edge_receivers = None

        self._check_shapes()
        self._check_device()

    """ =========== PROPERTIES =========== """

    @property
    def out_degree(self) -> torch.Tensor:
        """Compute out degree per node

        Returns:
            torch.Tensor: [n_node, n_dim*]
        """

        shape = [self.n_node, *self.edges.shape[1:]]
        cum_weights = torch.zeros(shape, device=self.senders.device)
        scatter_add(src=self.edges, index=self.senders, out=cum_weights, dim=0)
        return cum_weights

    @property
    def in_degree(self) -> torch.Tensor:
        """Compute in degree per node

        Returns:
            torch.Tensor: [n_node, n_dim*]
        """
        shape = [self.n_node, *self.edges.shape[1:]]
        cum_weights = torch.zeros(shape, device=self.device)
        scatter_add(
            src=self.edges, index=self.receivers, out=cum_weights, dim=0)
        return cum_weights

    @property
    def in_degree_counts(self):
        """Compute number of incoming edges per node

        Example:
            >>> import torch
            >>> graph = Graph(senders=torch.tensor([0, 1, 2, 3, 4, 5, 1]),
            ...               receivers=torch.tensor([1, 3, 5, 2, 1, 2, 5]),
            ...               nodes=torch.rand(6, 3),
            ...               edges=torch.rand(7),
            ...               n_node=6,
            ...               n_edge=7)
            >>> graph.in_degree_counts
            tensor([0, 2, 2, 1, 0, 2])

        Returns:
            torch.LongTensor: [n_node, ] count per node
        """
        return scatter_add(
            src=torch.ones(
                self.receivers.shape, dtype=torch.long, device=self.device),
            index=self.receivers,
            dim=0,
            dim_size=self.n_node)

    @property
    def out_degree_counts(self) -> torch.Tensor:
        """Compute number of outgoing edges per node

        Returns:
            torch.Tensor: [n_node, ]
        """
        return scatter_add(
            src=torch.ones(
                self.senders.shape, dtype=torch.long, device=self.device),
            index=self.senders,
            dim=0,
            dim_size=self.n_node)

    @property
    def pairwise_distances(self) -> torch.Tensor:
        """Compute pairwise distance (number of edges) between all pair of nodes
        Uses NetworkX shortest path algorithm, very slow for big graph.
        Caches the results.

        Returns:
            torch.Tensor: [n_node, n_node]
        """

        if self._distances is None:
            G = nx.DiGraph()
            G.add_edges_from(
                zip(numpify(self.senders), numpify(self.receivers)))
            G.add_nodes_from(range(self.n_node))
            self._distances = torch.zeros([self.n_node, self.n_node],
                                          device=self.device) - 1
            for source, targets in nx.shortest_path_length(G):
                for target, length in targets.items():
                    self._distances[source, target] = length

        return self._distances

    def edge(self, sender: int, receiver: int):
        """Return edge weight

        Args:
            sender (int): index of the sender node
            receiver (int): index of the receiver node

        Returns:
            torch.Tensor: [d_edge*, ]
        """
        mask = (self.senders == sender) & (self.receivers == receiver)
        edge = self.edges[mask].squeeze()
        return edge

    def dense_matrix(self) -> torch.Tensor:
        """Compute the dense adjacancy matrix. Only for 1D edge weight

        Returns:
            torch.Tensor: [n_node, n_node] weight matrix
        """
        edges = self.edges.squeeze()
        transition_matrix = torch.zeros([self.n_node, self.n_node],
                                        device=self.device)
        transition_matrix[self.senders, self.receivers] = edges
        return transition_matrix

    @property
    def coords(self) -> torch.Tensor:
        """Coordinates of the nodes
        Assumed to be the first 2 node features.

        Returns:
            torch.Tensor: [n_node, 2]
        """
        if self._coords is None:
            self.extract_coords_from_features()

        return self._coords

    def edge_vectors(self):
        """Returns vector from senders to receivers of each edge (assumes coords available)

        Returns:
            torch.Tensor: [n_edge, 2]
        """
        return self.coords[self.receivers] - self.coords[self.senders]

    def max_edge_weight_per_node(self) -> torch.Tensor:
        """Returns weight and edge_idx of the max weight outgoing edge for each node

        Returns:
            torch.Tensor: [n_node, ] weights, [n_node, ] indices
        """
        return scatter_max(self.edges.squeeze(), self.senders)

    def max_edge_vector_per_node(self) -> torch.Tensor:
        """Returns weighted vector per node in direction of highest weight edge"""
        weights, edge_ids = self.max_edge_weight_per_node()
        weights = weights - (1. / self.out_degree_counts.float())
        weights[self.out_degree_counts.float() == 0] = 0.
        return self.edge_vectors()[edge_ids] * weights.unsqueeze(1)

    def reverse_edges(self) -> 'Graph':
        """Returns new graph with senders/receivers inversed

        Returns:
            Graph: inversed graph
        """
        return self.update(senders=self.receivers, receivers=self.senders)

    def reorder_edges(self) -> 'Graph':
        """Reorder edges according to weight matrix entries
        Needed to be consistent with pyGSP `get_edge_list()` output.

        Returns:
            Graph:
        """
        # neet to have senders sorted, then receiver by senders sorted
        # use numpy stable sorting (mergesort for second sort to keep receivers ordered)
        senders = numpify(self.senders)
        receivers = numpify(self.receivers)

        indices = np.argsort(receivers)
        next_indices = np.argsort(senders[indices], kind='mergesort')
        new_indices = indices[next_indices]
        new_indices = torch.tensor(new_indices, device=self.device)

        return self.update(
            senders=self.senders[new_indices],
            receivers=self.receivers[new_indices],
            edges=self.edges[new_indices])

    def remove_self_loops(self) -> 'Graph':
        """Return new graph without self loops

        Returns:
            Graph: new graph
        """
        mask = self.senders == self.receivers
        return self.update(
            senders=self.senders[~mask],
            receivers=self.receivers[~mask],
            edges=self.edges[~mask],
            n_edge=(~mask).long().sum())

    def __matmul__(self, node_signal: torch.Tensor) -> torch.Tensor:
        """Compute multipication node_signal x W
        For each node, sum node_signal features from neighbors weighted
        by the edge between.

        Args:
            node_signal (torch.Tensor): value on the node

        Returns:
            torch.Tensor: new value on the nodes
        """

        assert node_signal.shape[0] == self.n_node
        assert self.edges is not None and \
            self.edges.squeeze().dim() == 1

        senders_features = node_signal[self.senders]  # n_edge x d_node
        broadcast_edges = self.edges.view(
            -1, *([1] * (node_signal.dim() - 1)))  # n_edge, 1 ... 1
        weighted_senders = senders_features * broadcast_edges  # n_edge x d_node

        node_results = scatter_add(
            src=weighted_senders,
            index=self.receivers,
            dim=0,
            dim_size=self.n_node)
        return node_results

    def add_self_loops(self,
                       edge_value: float = 1.,
                       degree_zero_only: bool = False) -> 'Graph':
        """Add self loops to nodes
            edge_value (float, optional): Defaults to 1.. Value for the added edges
            degree_zero_only (bool, optional): Defaults to False. Add self-loops only to degree 0 nodes

        Returns:
            Graph: new graph
        """

        if degree_zero_only:
            add_self_loop_nodes = (self.out_degree_counts == 0).nonzero()[:, 0]
        else:
            add_self_loop_nodes = torch.arange(self.n_node, device=self.device)

        new_senders = torch.cat([self.senders, add_self_loop_nodes])
        new_receivers = torch.cat([self.receivers, add_self_loop_nodes])
        new_edges = torch.cat([
            self.edges, edge_value * torch.ones(
                [len(add_self_loop_nodes), *self.edges.shape[1:]],
                device=self.device)
        ])

        return self.update(
            senders=new_senders,
            receivers=new_receivers,
            edges=new_edges,
            n_edge=self.n_edge + len(add_self_loop_nodes))

    def normalize_weights(self) -> 'Graph':
        """Normalize outgoing weight, sum of outgoing edges is 1.

        Returns:
            Graph: new Graph
        """
        new_edges = self.edges / self.out_degree[self.senders]
        return self.update(edges=new_edges)

    def softmax_weights(self):
        """Compute the softmax of the outgoing edge weights by node.
        Use the shift property of softmax for stability.

        Returns:
            Graph: new Graph
        """
        max_out_weight_per_node, _ = scatter_max(
            src=self.edges,
            index=self.senders,
            dim=0,
            dim_size=self.n_node,
            fill_value=-1e20)
        shifted_weights = self.edges - max_out_weight_per_node[self.senders]

        exp_weights = shifted_weights.exp()
        normalizer = scatter_add(
            src=exp_weights, index=self.senders, dim=0, dim_size=self.n_node)
        sender_normalizer = normalizer[self.senders]
        normalized_weights = exp_weights / sender_normalizer

        if any_nan(normalized_weights):
            logging.warning(
                "NaN weight after normalization in graph `softmax_weights`")

        return self.update(edges=normalized_weights)

    def extract_coords_from_features(self, keep_in_features: bool = True):
        """Extract the first two node features as ploting coordinates
            keep_in_features (bool, optional): Defaults to True. Remove the coordinates from the node features or not
        """
        assert self.nodes is not None and self.nodes.shape[1] >= 2, \
            "Nodes coordinates are missing first 2 node features"

        self._coords = self.nodes[:, :2]

        if not keep_in_features:
            if self.nodes.shape[1] == 2:
                self.nodes = None
            else:
                self.nodes = self.nodes[:, 2:]

    def edge_features_with_nodes(self) -> torch.Tensor:
        """For each edge stack edge features, sender features and receiver features

        Returns:
            torch.Tensor: [n_edge, d_edge + d_node * 2]
        """

        if self._augmented_edge_features is None:
            features = []
            if self.edges is not None:
                features.append(self.edges)
            if self.nodes is not None:
                features.append(self.nodes[self.senders])
            if self.nodes is not None:
                features.append(self.nodes[self.receivers])
            self._augmented_edge_features = torch.cat(
                [f.view(self.n_edge, -1) for f in features], dim=-1)

        return self._augmented_edge_features

    """ -------- RANDOM WALKS -------- """

    def sample_random_walks(self,
                            start_node: Union[int, torch.Tensor],
                            num_samples: int,
                            num_steps: int,
                            allow_backward: bool = True):
        """Sample some random walk on the graph starting at `start_node`

        Args:
            start_node (Union[int, torch.Tensor]): starting node index (int) or probablity distribution over the nodes
            num_samples (int): number of sample path to draw
            num_steps (int): number of steps for each sample
            allow_backward (bool, optional): Defaults to True. Allow to go back to right previous nodes

        Returns:
            (torch.LongTensor, torch.LongTensor): traversed nodes, traversed edge ids [num_samples, num_steps + 1], [num_samples, num_steps]
            if stuck in deadend return -1 indices
        """
        assert_allclose(
            self.out_degree, 1., message='Graph should be a random walk graph')

        start_nodes = torch.zeros(
            num_samples, device=self.device, dtype=torch.long)
        if type(start_node) is int or (type(start_node) is torch.Tensor
                                       and start_node.dim() == 0):
            start_nodes[:] = start_node
        else:
            for i in range(num_samples):
                start_nodes[i] = sample(
                    torch.arange(len(start_node)), start_node)

        traversed_nodes = torch.zeros([num_samples, num_steps + 1],
                                      device=self.device,
                                      dtype=torch.long) - 1
        traversed_nodes[:, 0] = start_nodes
        traversed_edges = torch.zeros(
            [num_samples, num_steps], device=self.device, dtype=torch.long) - 1

        for i_sample in range(num_samples):
            curr_node = start_nodes[i_sample]
            for step in range(num_steps):
                possible_edges_mask = self.senders == curr_node
                if not allow_backward and step >= 1:
                    possible_edges_mask &= (
                        self.receivers != traversed_nodes[i_sample, step - 1])

                if possible_edges_mask.long().sum() == 0:
                    break  # dead end random walk

                edge_ids = possible_edges_mask.nonzero()[:, 0]
                taken_edge = sample(
                    edge_ids,
                    self.edges[edge_ids] / self.edges[edge_ids].sum())
                curr_node = self.receivers[taken_edge]

                traversed_edges[i_sample, step] = taken_edge
                traversed_nodes[i_sample, step + 1] = curr_node

        return traversed_nodes, traversed_edges

    def random_walk(self, start_nodes: torch.Tensor,
                    num_steps: int) -> torch.Tensor:
        """Take a random walk for num_steps steps
        Args:
            start_nodes (torch.Tensor): [n_node,] probability distribution
            num_steps (int): number of steps to take

        Returns:
            torch.Tensor: result distribution
        """
        if num_steps == 0:
            return start_nodes

        node_signal = start_nodes
        for _ in range(num_steps):
            node_signal = self @ node_signal
        return node_signal

    def compute_non_backtracking_edges(
            self) -> (torch.LongTensor, torch.LongTensor):
        """Compute non backtracking possible edge transitions indices

        We cache `edge_senders` and `edge_receivers` to recompute for different weights
        The computation done on CPU otherwise out of memory on GPU

        Returns:
            (torch.LongTensor, torch.LongTensor): edge senders, edge receivers
        """
        if self._non_backtracking_edge_senders is None or self._non_backtracking_edge_receivers is None:
            senders, receivers = self.senders.to("cpu"), self.receivers.to(
                "cpu")
            continuing_edges = receivers.unsqueeze(1) == senders.unsqueeze(0)
            looping_edges = senders.unsqueeze(1) == receivers.unsqueeze(0)
            non_backtracking_edges = continuing_edges & ~looping_edges

            nz = non_backtracking_edges.nonzero().to(self.device)
            edge_senders = nz[:, 0]
            edge_receivers = nz[:, 1]
            self._non_backtracking_edge_senders, self._non_backtracking_edge_receivers = edge_senders, edge_receivers

        return self._non_backtracking_edge_senders, self._non_backtracking_edge_receivers

    @property
    def non_backtracking_random_walk_graph(self) -> 'Graph':
        """Create an edge graph with only continuous non looping pairs of edges
        Self must be a random walk graph on nodes.

        See https://scholar.harvard.edu/files/mkempton/files/nb_walk_paper.pdf

        Returns:
            Graph: Non backtracking random walk graph
        """

        if self._non_backtracking_random_walk_graph is None:
            edge_senders, edge_receivers = self.compute_non_backtracking_edges(
            )

            assert_allclose(
                self.out_degree,
                1.,
                message=
                "Graph to be transformed in non backtracking random walk graph should be a random walk graph"
            )
            edge_weights = self.edges[edge_receivers]

            G = Graph(
                senders=edge_senders,
                receivers=edge_receivers,
                nodes=None,
                edges=edge_weights,
                n_node=self.n_edge,
                n_edge=len(edge_weights))

            G = G.add_self_loops(degree_zero_only=True)  # for deadends
            G = G.normalize_weights()
            self._non_backtracking_random_walk_graph = G

        return self._non_backtracking_random_walk_graph

    def non_backtracking_random_walk(self, start_nodes: torch.Tensor,
                                     num_steps: int) -> torch.Tensor:
        """Take a random walk on the non backtracking graph

        First step is taken from node to edge values (no illegal backtracking)
        Following (n-1) steps are taken on the edge graph
        Finally we go back to nodes by summing incoming edge values

        Args:
            start_nodes (torch.Tensor): [n_node,] probaiblity distribution
            num_steps (int): number of steps to take

        Returns:
            torch.Tensor: result distribution
        """
        if num_steps == 0:
            return start_nodes

        edge_start = start_nodes[self.senders] * self.edges
        edge_signal = edge_start
        for _ in range(num_steps - 1):
            edge_signal = self.non_backtracking_random_walk_graph @ edge_signal

        node_signal = scatter_add(
            src=edge_signal, index=self.receivers, dim_size=self.n_node)
        return node_signal

    """ ======= PLOTTING ======="""

    def plot_signal(self, *args, **kwargs):
        """Calls pygsp.plot_signal with numpified torch.Tensor arguments"""
        return self.pygsp.plot_signal(
            *[numpify(a) for a in args],
            **{k: numpify(v)
               for k, v in kwargs.items()})

    def plot(self, *args, **kwargs):
        """Calls pygsp.plot with numpified torch.Tensor arguments"""
        return self.pygsp.plot(*[numpify(a) for a in args],
                               **{k: numpify(v)
                                  for k, v in kwargs.items()})

    def plot_trajectory(self,
                        distributions: torch.Tensor,
                        colors: list,
                        with_edge_arrows: bool = False,
                        highlight: Union[int, List[int]] = None,
                        zoomed: bool = False,
                        ax=None,
                        normalize_intercept: bool = False,
                        edge_width: float = .1):
        """Plot a trajectory on this graph

        Args:
            distributions (torch.Tensor): [n_observations, n_node] sequence of probability distribution to plot
            colors (list): [n_observations, ] color per observation
            with_edge_arrows (bool, optional): Defaults to False. Show strongest edge direction arrow at each node
            highlight (Union[int, List[int]], optional): Defaults to None. Some node id(s) to highlight
            zoomed (bool, optional): Defaults to False. Zoom only onto the interesting part of the distributions (not too small)
            ax (optional): Defaults to None. Matplotlib axis
            normalize_intercept (bool, optional): Defaults to False. PyGSP plotting normalize intercept for widths
            edge_width (float, optional): Defaults to .1. edge width

        Returns:
            fig, ax from matplotlib
        """

        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(111)

        if zoomed:
            display_points_mask = distributions.sum(dim=0) > 1e-4
            display_coords = self.coords[display_points_mask]
            xmin, xmax = display_coords[:, 0].min(), display_coords[:, 0].max()
            ymin, ymax = display_coords[:, 1].min(), display_coords[:, 1].max()
            xcenter, ycenter = (xmax + xmin) / 2, (ymax + ymin) / 2
            size = max(xmax - xmin, ymax - ymin)
            margin = size * 1.1 / 2
            ax.set_xlim([xcenter - margin, xcenter + margin])
            ax.set_ylim([ycenter - margin, ycenter + margin])

        # plot underlying edges
        vertex_size = 0.
        if highlight is not None:
            vertex_size = np.zeros(self.n_node)
            vertex_size[highlight] = .5

        # HACK in pygsp.plotting, remove alpha at lines 533 and 541
        self.pygsp.plotting['highlight_color'] = gnx_plot.green
        self.pygsp.plotting['normalize_intercept'] = 0.
        self.plot(
            edge_width=edge_width,  # highlight=highlights,
            edges=True,
            vertex_size=vertex_size,  # transparent nodes
            vertex_color=[(0., 0., 0., 0.)] * self.n_node,
            highlight=highlight,
            ax=ax)

        # plot distributions
        transparent_colors = [mpl.colors.to_hex(mpl.colors.to_rgba(c, alpha=.5), keep_alpha=True) for c in colors]

        self.pygsp.plotting['normalize_intercept'] = 0.
        for distribution, color in zip(distributions, transparent_colors):
            self.plot(
                vertex_size=distribution,
                vertex_color=color,
                edge_width=0,
                ax=ax)

        if with_edge_arrows:
            coords = self.coords
            arrows = self.max_edge_vector_per_node()
            coords = numpify(coords)
            arrows = numpify(arrows)
            ax.quiver(
                coords[:, 0],
                coords[:, 1],
                arrows[:, 0],
                arrows[:, 1],
                pivot='tail')

        ax.set_aspect('equal')
        return ax

    """ ======= READ/WRITE/EXPORT/IMPORT ======="""

    @property
    def pygsp(self) -> pygsp.graphs.Graph:
        """Create a PyGSP graph from this graph

        Returns:
            pygsp.graphs.Graph: the PyGSP graph
        """

        if self._pygsp is None:
            weights = self.edges.squeeze()
            assert weights.dim() == 1
            weights = weights.detach().to('cpu').numpy()
            senders = self.senders.detach().to('cpu').numpy()
            receivers = self.receivers.detach().to('cpu').numpy()

            W = scipy.sparse.coo_matrix((weights, (senders, receivers)))
            coords = self.coords.detach().to('cpu').numpy()
            self._pygsp = pygsp.graphs.Graph(W, coords=coords)

        return self._pygsp

    @classmethod
    def from_pygsp_graph(cls, G) -> 'Graph':
        """Construct a Graph from a PyGSP graph

        Returns:
            Graph: new graph
        """

        senders, receivers, weights = map(torch.tensor, G.get_edge_list())
        senders = senders.long()
        receivers = receivers.long()
        edges = weights.float()
        n_edges = G.n_edges

        # consider directed graph
        if not G.is_directed():
            senders, receivers = torch.cat([senders, receivers]), torch.cat(
                [receivers, senders])
            edges = torch.cat([edges, edges])
            n_edges *= 2

        nodes = torch.tensor(
            G.coords).float() if G.coords is not None else None

        return Graph(
            senders=senders,
            receivers=receivers,
            edges=edges,
            nodes=nodes,
            n_node=G.n_vertices,
            n_edge=n_edges)

    @classmethod
    def read_from_files(cls, nodes_filename: str,
                        edges_filename: str) -> 'Graph':
        """
        Load a graph from files `nodes.txt` and 'edges.txt`

        Node file starts with number of nodes, number of features per node
        Followed by one line per node, id then features. Example:
        ```
        18156	2
        0	6.6491811	46.5366765
        1	6.6563029	46.5291637
        2	6.6488104	46.5365551
        3	6.6489423	46.5367163
        4	6.649007	46.5366124
        5	6.5954845	46.5224695
        ...
        ```

        Edge file starts with number of edges, number of features per edges
        Followed by one line per edge: id, from_node, to_node, then features. Example:

        ```
        32468	2
        0	0	6	11.495	50
        1	1	10517	23.887	20
        2	1	10242	8.34	20
        3	2	4	16.332	50
        4	2	11342	13.31	-1
        5	2	6439	15.761	50
        6	2	11344	15.797	50
        ```
        """
        node_features = None
        edge_features = None

        # read node features
        with open(nodes_filename) as f:
            num_nodes, num_node_features = map(int, f.readline().split('\t'))
            if num_node_features > 0:
                node_features = torch.zeros(num_nodes, num_node_features)
                for i, line in enumerate(f.readlines()):
                    features = torch.tensor(
                        list(map(float,
                                 line.split('\t')[1:])))
                    node_features[i] = features

        # read edge features
        with open(edges_filename) as f:
            num_edges, num_edge_features = map(int, f.readline().split('\t'))

            senders = torch.zeros(num_edges, dtype=torch.long)
            receivers = torch.zeros(num_edges, dtype=torch.long)

            if num_edge_features > 0:
                edge_features = torch.zeros(num_edges, num_edge_features)

            for i, line in enumerate(f.readlines()):
                elements = line.split('\t')
                senders[i] = int(elements[1])
                receivers[i] = int(elements[2])
                if edge_features is not None:
                    edge_features[i] = torch.tensor(
                        list(map(float, elements[3:])))

        return Graph(
            nodes=node_features,
            edges=edge_features,
            senders=senders,
            receivers=receivers,
            n_node=num_nodes,
            n_edge=num_edges)

    def write_to_directory(self, directory: str):
        """Write `nodes.txt` and 'edges.txt` into `directory`
        See `read_from_files` method documentation for the format
        """
        os.makedirs(directory, exist_ok=True)

        with open(os.path.join(directory, 'nodes.txt'), 'w') as f:
            f.write("{}\t{}\n".format(
                self.n_node, 0 if self.nodes is None else self.nodes.shape[1]))
            if self.nodes is not None:
                for i, features in enumerate(self.nodes):
                    line = str(i) + "\t" + "\t".join(
                        map(str, [f.item() for f in features])) + "\n"
                    f.write(line)

        edges = self.edges
        if edges is not None and edges.dim() == 1:
            edges = edges.unsqueeze(-1)
        if edges is None:
            edges = [[]] * self.n_edge

        with open(os.path.join(directory, 'edges.txt'), 'w') as f:
            f.write("{}\t{}\n".format(
                self.n_edge, 0 if self.edges is None else edges.shape[1]))
            for i, (sender, receiver, features) in enumerate(
                    zip(self.senders, self.receivers, edges)):
                line = "\t".join(map(str, [i, sender.item(), receiver.item()])) + \
                       "\t" + \
                       "\t".join(map(str, [f.item() for f in features])) + "\n"
                f.write(line)

    @classmethod
    def from_networkx_graph(graph,
                            node_feature_field: str = 'feature',
                            edge_feature_field: str = 'feature') -> 'Graph':
        """Create Graph from a NetworkX graph
        """
        g = nx.convert_node_labels_to_integers(graph)
        n_node = g.number_of_nodes()
        n_edge = g.number_of_edges()
        senders = torch.tensor([e[0] for e in g.edges()])
        receivers = torch.tensor([e[1] for e in g.edges()])

        nodes = None
        if n_node > 0 and node_feature_field in g.nodes[0]:
            shape = [
                n_node, *torch.tensor(g.nodes[0][node_feature_field]).shape
            ]
            if len(shape) == 1:
                shape = [shape[0], 1]
            nodes = torch.zeros(shape).float()
            for i in range(n_node):
                nodes[i] = torch.tensor(g.nodes[i][node_feature_field])

        edges = None
        if n_edge > 0:
            first_edge_data = next(iter(g.edges(data=True)))[2]
            if edge_feature_field in first_edge_data:
                shape = [
                    n_edge,
                    *torch.tensor(first_edge_data[edge_feature_field]).shape
                ]
                if len(shape) == 1:
                    shape = [shape[0], 1]
                edges = torch.zeros(shape).float()
                for i, (_, _, features) in enumerate(
                        g.edges.data(edge_feature_field)):
                    edges[i] = torch.tensor(features)

        if not g.is_directed():
            if edges is not None:
                edges = torch.cat([edges, edges])
            senders, receivers = torch.cat([senders, receivers]), torch.cat(
                [receivers, senders])

        g = Graph(
            nodes=nodes,
            edges=edges,
            receivers=receivers,
            senders=senders,
            n_node=n_node,
            n_edge=n_edge)

        return g

    """ ======= GRAPH SEMANTIC ======="""

    def __repr__(self):
        nodes_str = None if self.nodes is None else list(self.nodes.shape)
        edges_str = None if self.edges is None else list(self.edges.shape)
        return f"Graph(n_node={self.n_node}, n_edge={self.n_edge}, nodes={nodes_str}, edges={edges_str})"

    def clone(self) -> 'Graph':
        """Shallow copy"""
        return copy.copy(self)

    def update(self, **kwargs) -> 'Graph':
        """Create a copy of this graph with updated fields

        Args:
            kwargs: fields and values to update

        Returns:
            Graph: the new graph
        """
        for k in kwargs:
            if k[0] == "_":
                raise ValueError(
                    f"Graph update should not affect _protected attribute '{k}'"
                )

        g = self.clone()

        # update the fields
        for k, v in kwargs.items():
            setattr(g, k, v)

        # remove precomputed fields that need to be recomputed
        if any(k in ["receivers", "senders"] for k in kwargs):
            g._non_backtracking_edge_senders = None
            g._non_backtracking_edge_receivers = None

        if any(k in ["receivers", "senders", "edges"] for k in kwargs):
            g._non_backtracking_random_walk_graph = None

        if any(k in ["receivers", "senders"] for k in kwargs):
            g._distances = None

        if any(k in ["receivers", "senders", "nodes", "edges"]
               for k in kwargs):
            g._pygsp = None

        g._check_device()
        g._check_shapes()
        return g

    def to(self, device: torch.device) -> 'Graph':
        """Move this Graph instance to the required device

        Returns:
            Graph: moved Graph
        """
        if self.device == device:
            return self

        moved_graph = self.clone()
        moved_graph.device = device
        for attribute, value in moved_graph.__dict__.items():
            if type(value) is torch.Tensor or type(value) is Graph:
                moved_graph.__dict__[attribute] = value.to(device)
        return moved_graph

    def _check_device(self):
        """Check that all attributes of the graph are on `self.device`

        Raises:
            ValueError: if a tensor is not on the right device
        """
        for attribute, value in self.__dict__.items():
            if hasattr(value, 'device') and value.device != self.device:
                raise ValueError(
                    f"Graph attribute '{attribute}' is on device '{value.device}' instead of '{self.device}'"
                )

    def _check_shapes(self):
        def check_not_none(value, name: str):
            if value is None:
                raise ValueError(f"Graph field '{name}' should not be None")

        check_not_none(self.n_node, 'n_node')
        check_not_none(self.n_edge, 'n_edge')

        if self.nodes is not None and self.nodes.shape[0] != self.n_node:
            raise ValueError(
                f"Nodes feature tensor should have the first dimension of size `n_node` ({self.nodes.shape[0]} instead of {self.n_node})"
            )

        if self.edges is not None:
            if self.senders.dim() != 1:
                raise ValueError("Graph `senders` should be 1D")

            if self.receivers.dim() != 1:
                raise ValueError("Graph `receivers` should be 1D")

            if self.edges.shape[0] != self.n_edge or \
                self.senders.shape[0] != self.n_edge or \
                self.receivers.shape[0] != self.n_edge:
                raise ValueError(f"Incorrect Graph `edges` shape")
