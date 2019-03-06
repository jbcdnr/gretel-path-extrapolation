import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_add
from typing import Optional

from graph import Graph


class MLP(nn.Module):
    def __init__(self, d_in: int, d_out: int):
        """Two layer perception with sigmoid non linearity

        Args:
            d_in (int): input features size
            d_out (int): output feautres size
        """
        super(MLP, self).__init__()
        self.d_in = d_in
        self.d_out = d_out
        self.fc1 = nn.Linear(d_in, d_in * 2)
        self.fc2 = nn.Linear(d_in * 2, d_out)

    def forward(self, X):
        return self.fc2(torch.sigmoid(self.fc1(X)))


class Model(nn.Module):
    def __init__(
        self,
        diffusion_graph_transformer: Optional["EdgeTransformer"],
        multichannel_diffusion: Optional["MultiDiffusion"],
        direction_edge_mlp: Optional[MLP],
        number_observations: int,
        rw_expected_steps: int,
        rw_non_backtracking: bool,
        latent_transformer_see_target: bool,
        double_way_diffusion: bool,
        diffusion_self_loops: bool,
    ):
        """The Gretel

        Args:
            diffusion_graph_transformerOptional (Optional[EdgeTransformer]):
                module for computing edge weight for the diffusion
                default: take 1 / out_degree for each edge weight
            multichannel_diffusion (Optional[MultiDiffusion]):
                module that computes the diffusion of the past observation on the graph (outputs virutal coordinates)
                default: ignore virtual coordinates
            direction_edge_mlp (Optional[MLP]):
                module that compute the latent graph edge weights
                default: take 1 / out_degree for each edge weight
            number_observations (int): number of observations seen in the past of the trajectories
            rw_expected_steps (int): number of steps to be taken by the path generator (-1 if adaptive)
            rw_non_backtracking (bool): use non backtracking generator
            latent_transformer_see_target (bool): show node cross features of target as input to direction_edge_mlp
            double_way_diffusion (bool): multichannel_diffusion is run on the graph and reversed graph (reversed edge direction)
            diffusion_self_loops (bool): add self loop edges to all nodes on the diffusion graph

        Shapes:
            observations: [traj_length, n_node]
            graph: Graph
            observed: [n_pred, number_observations]
            starts: [n_pred, ] start indices in observations
            targets: [n_pred, ] target indices in observations
            pairwise_node_features: [n_node, n_node]
            number_steps: [traj_length - 1, ] edge distance between consecutive observations (used in random walk)
        """
        super(Model, self).__init__()

        # params
        self.number_observations = number_observations
        self.rw_expected_steps = rw_expected_steps
        self.rw_non_backtracking = rw_non_backtracking
        self.latent_transformer_see_target = latent_transformer_see_target
        self.double_way_diffusion = double_way_diffusion
        self.diffusion_self_loops = diffusion_self_loops

        # modules
        self.diffusion_graph_transformer = diffusion_graph_transformer
        self.multichannel_diffusion = multichannel_diffusion
        self.direction_edge_mlp = direction_edge_mlp

    def forward(
        self,
        observations,
        graph: Graph,
        diffusion_graph: Graph,
        observed,
        starts,
        targets,
        pairwise_node_features,
        number_steps=None,
    ):
        # check shapes
        assert observed.shape[0] == starts.shape[0] == targets.shape[0]
        n_pred = observed.shape[0]

        # baseline
        if self.diffusion_graph_transformer is None and self.direction_edge_mlp is None:
            # if not a random graph, take the uniform random graph
            if (
                graph.edges.shape != torch.Size([graph.n_edge, 1])
                or ((graph.out_degree - 1.0).abs() > 1e-5).any()
            ):
                rw_graphs = graph.update(
                    edges=torch.ones([graph.n_edge, n_pred], device=graph.device)
                )
                rw_graphs = rw_graphs.softmax_weights()
            else:
                rw_graphs = graph
            virtual_coords = None
        else:
            # compute diffusions
            virtual_coords = self.compute_diffusion(diffusion_graph, observations)
            if self.double_way_diffusion:
                virtual_coords_reversed = self.compute_diffusion(
                    diffusion_graph.reverse_edges(), observations
                )
                virtual_coords = torch.cat([virtual_coords, virtual_coords_reversed])

            # compute rw graph
            rw_graphs = self.compute_rw_weights(
                virtual_coords, observed, pairwise_node_features, targets, graph
            )

        # random walks
        target_distributions = self.compute_random_walk(
            rw_graphs, observations, starts, targets, number_steps
        )
        rw_weights = rw_graphs.edges.transpose(0, 1)

        return target_distributions, virtual_coords, rw_weights

    def compute_diffusion(self, graph, observations) -> torch.Tensor:
        if self.diffusion_self_loops:
            graph = graph.add_self_loops()

        if self.diffusion_graph_transformer:
            diffusion_graph = self.diffusion_graph_transformer(graph)
        else:
            diffusion_graph = graph.update(edges=torch.ones([graph.n_edge, 1], device=graph.device))

        diffusion_graph = diffusion_graph.softmax_weights()

        # run the diffusion for each observation
        diffusion_graph = diffusion_graph.update(
            nodes=observations.t()
        )  # n_node x trajectory_length
        virtual_coords = self.multichannel_diffusion(
            diffusion_graph
        )  # n_node x trajectory_length x hidden

        return virtual_coords

    def compute_rw_weights(
        self, virtual_coords, observed, pairwise_node_features, targets, graph: Graph
    ) -> Graph:
        n_pred = observed.shape[0]
        witness_features = []

        # visible diffusions
        # -- n_node x batch x (number_observations * hidden)
        diffusions = virtual_coords[:, observed].view(graph.n_node, n_pred, -1)
        witness_features.append(diffusions[graph.senders])
        witness_features.append(diffusions[graph.receivers])

        # original node features
        # -- n_node x batch x d_node
        if graph.nodes is not None:
            nodes = graph.nodes.view(graph.n_node, 1, -1).repeat(1, n_pred, 1)
            witness_features.append(nodes[graph.senders])
            witness_features.append(nodes[graph.receivers])

        # pairwise node-target features
        # -- n_node x batch x d_cross
        if self.latent_transformer_see_target and pairwise_node_features is not None:
            target_features = pairwise_node_features[targets].transpose(0, 1)
            witness_features.append(target_features[graph.senders])
            witness_features.append(target_features[graph.receivers])

        # original edge features
        # n_edge x batch x d_edge
        if graph.edges is not None:
            witness_features.append(graph.edges.view(graph.n_edge, 1, -1).repeat(1, n_pred, 1))

        # target features
        # -- n_edge x batch x d_node
        if self.latent_transformer_see_target and pairwise_node_features is not None:
            witness_features.append(graph.nodes[targets].unsqueeze(0).repeat(graph.n_edge, 1, 1))

        # -- n_edge x (...)
        edge_input = torch.cat(witness_features, dim=2)
        edge_input = edge_input.view(n_pred * graph.n_edge, -1)
        rw_weights = self.direction_edge_mlp(edge_input).view(graph.n_edge, -1)

        rw_graphs = graph.update(edges=rw_weights)
        rw_graphs = rw_graphs.softmax_weights()

        return rw_graphs

    def compute_random_walk(
        self, rw_graphs, observations, starts, targets, number_steps
    ) -> torch.Tensor:
        n_pred = len(starts)
        n_node = observations.shape[1]
        device = observations.device
        rw_weights = rw_graphs.edges.transpose(0, 1)

        start_distributions = observations[starts]  # batch x n_node
        rw_steps = self.compute_number_steps(starts, targets, number_steps)

        predict_distributions = torch.zeros(n_pred, n_node, device=device)

        for pred_id in range(n_pred):
            rw_graph = rw_graphs.update(edges=rw_weights[pred_id])

            max_step_rw = None
            if self.rw_expected_steps:
                max_step_rw = rw_steps[pred_id]

            start_nodes = start_distributions[pred_id]
            if self.rw_non_backtracking:
                predict_distributions[pred_id] = rw_graph.non_backtracking_random_walk(
                    start_nodes, max_step_rw
                )
            else:
                predict_distributions[pred_id] = rw_graph.random_walk(start_nodes, max_step_rw)

        return predict_distributions

    @staticmethod
    def compute_number_steps(starts, targets, number_steps):
        if number_steps is None:
            return None

        cum_num_steps = torch.cat(
            [torch.tensor([0], device=number_steps.device), torch.cumsum(number_steps, dim=0)]
        )
        return cum_num_steps[targets] - cum_num_steps[starts]


class EdgeTransformer(nn.Module):
    def __init__(self, d_node: int, d_edge: int, d_edge_out: int, non_linearity=torch.sigmoid):
        """Transformer of edge weights

        2 layer perceptron to compute edge weight from sender/receiver node features and edge features

        Args:
            d_node (int): node dimension
            d_edge (int): edge dimension
            d_edge_out (int): edge output dimension
            non_linearity (optional): Defaults to torch.sigmoid. non linearity

        Shapes:
            graph.nodes [n_node, d_node]
            graph.edges [n_edge, d_edge]
            output.edges [n_edge, d_edge_out]
        """

        super(EdgeTransformer, self).__init__()
        d = 2 * d_node + d_edge
        self.fc1 = nn.Linear(d, 2 * d)
        self.fc2 = nn.Linear(2 * d, d_edge_out)
        self.non_linearity = non_linearity

    def forward(self, graph):
        in_features = []
        if graph.nodes is not None:
            nodes = graph.nodes.view(graph.n_node, -1)
            in_features.append(nodes[graph.senders])
            in_features.append(nodes[graph.receivers])
        if graph.edges is not None:
            edges = graph.edges.view(graph.n_edge, -1)
            in_features.append(edges)

        in_features = torch.cat(in_features, dim=-1)
        new_edges = self.fc2(self.non_linearity(self.fc1(in_features)))
        return graph.update(edges=new_edges)


class MultiDiffusion(nn.Module):
    def __init__(self, diffusion_k_hops, diffusion_hidden_dimension, parametrized_diffusion=True):
        """Applies parametrized mutli channel diffusions

        Args:
            diffusion_k_hops (int): number of hops of diffusion
            diffusion_hidden_dimension (int): dimension of the node feature during diffusion
            parametrized_diffusion (bool): learn weight for layers and non linearity

        Shape:
            graph: Graph
                edges (n_edge, 1)
                nodes (n_node, num_observations)

        Returns:
            (n_node, num_observations, diffusion_hidden_dimension)
        """
        super(MultiDiffusion, self).__init__()
        self.diffusion_hidden_dimension = diffusion_hidden_dimension
        self.diffusion_k_hops = diffusion_k_hops
        self.parametrized_diffusion = parametrized_diffusion

        if parametrized_diffusion:
            self.lift = nn.Linear(1, self.diffusion_hidden_dimension)
            self.layers = nn.ModuleList(
                [nn.Linear(self.diffusion_hidden_dimension, self.diffusion_hidden_dimension)]
                * self.diffusion_k_hops
            )

        if not parametrized_diffusion:
            assert diffusion_hidden_dimension == 1

    def forward(self, graph):
        # -- (n_node, num_observations, 1)
        X = graph.nodes.unsqueeze(-1)

        if not self.parametrized_diffusion:
            for _ in range(self.diffusion_k_hops):
                X = graph @ X
        else:
            # -- (n_node, num_observations, diffusion_hidden_dimension)
            X = self.lift(X)
            for layer in self.layers:
                X = graph @ X
                X = layer(X)
                X = F.relu(X)

        # -- (n_node, num_observations, diffusion_hidden_dimension)
        return X

