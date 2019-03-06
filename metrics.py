from collections import defaultdict, OrderedDict

import torch
from torch_scatter import scatter_max
from tqdm import tqdm
from tabulate import tabulate

from utils import generate_masks
from trajectories import Trajectories
from graph import Graph
from model import Model


class Metric:
    """Simple Metric aggregation class"""

    def __init__(self):
        self.sum = 0.0
        self.count = 0

    def reset(self):
        self.sum = 0.0
        self.count = 0

    def add(self, v):
        if type(v) is torch.Tensor:
            assert v.dim() == 0
            v = v.float().item()
        self.sum += float(v)
        self.count += 1

    def add_all(self, vs):
        self.sum += vs.sum().float().item()
        self.count += len(vs)

    def mean(self):
        return self.sum / self.count


class Evaluator:
    """Compute the metrics for some trajectories predictions"""

    def __init__(self, n_node, given_as_target=None, siblings_nodes=None, config=None):
        self.given_as_target = given_as_target
        self.siblings_nodes = siblings_nodes
        self.config = config

        self.metrics = OrderedDict()
        self.metrics_by_key = OrderedDict()

        if config.dataset == "wikispeedia":
            harmonic_numbers = torch.arange(n_node, device=config.device).float() + 1
            harmonic_numbers = 1.0 / harmonic_numbers
            harmonic_numbers = torch.cumsum(harmonic_numbers, dim=0)
            self.harmonic_numbers = harmonic_numbers  # CAREFUL: indexed from 0

    def init_metrics(self):
        metrics = [
            "target_probability",
            "choice_accuracy",
            "choice_accuracy_deg3",
            "precision_top1",
            "precision_top5",
            "path_nll",
            "path_nll_deg3",
        ]
        self.metrics = OrderedDict()
        for m in metrics:
            self.metrics[m] = Metric()

        if self.config.dataset == "wikispeedia":
            self.metrics_by_key = OrderedDict()
            metrics_by_key = ["precision_top1", "precision_top5", "target2_acc", "target_rank"]
            for m in metrics_by_key:
                self.metrics_by_key[m] = defaultdict(lambda: Metric())

    def compute(
        self,
        model: Model,
        graph: Graph,
        trajectories: Trajectories,
        pairwise_features: torch.Tensor,
    ):
        """Update the metrics for all trajectories in `trajectories`"""
        self.init_metrics()
        config = self.config

        with torch.no_grad():
            for trajectory_idx in tqdm(range(len(trajectories))):
                observations = trajectories[trajectory_idx]

                number_steps = None
                if config.rw_edge_weight_see_number_step or config.rw_expected_steps:
                    if config.use_shortest_path_distance:
                        number_steps = (
                            trajectories.leg_shortest_lengths(trajectory_idx).float() * 1.1
                        ).long()
                    else:
                        number_steps = trajectories.leg_lengths(trajectory_idx)

                observed, starts, targets = generate_masks(
                    trajectory_length=observations.shape[0],
                    number_observations=config.number_observations,
                    predict=config.target_prediction,
                    with_interpolation=config.with_interpolation,
                    device=config.device,
                )

                diffusion_graph = (
                    graph if not config.diffusion_self_loops else graph.add_self_loops()
                )

                predictions, _, rw_weights = model(
                    observations,
                    graph,
                    diffusion_graph,
                    observed=observed,
                    starts=starts,
                    targets=targets,
                    pairwise_node_features=pairwise_features,
                    number_steps=number_steps,
                )

                self.update_metrics(
                    trajectories,
                    graph,
                    observations,
                    observed,
                    starts,
                    targets,
                    predictions,
                    rw_weights,
                    trajectory_idx,
                    model.rw_non_backtracking,
                )

    def get_metrics(self):
        metrics = OrderedDict()
        for m, v in self.metrics.items():
            metrics[m] = v.mean()

        metrics_by_key = OrderedDict()
        for m, vs in self.metrics_by_key.items():
            metrics_by_key[m] = OrderedDict()
            for k in sorted(vs.keys()):
                metrics_by_key[m][k] = vs[k].mean()

        return metrics, metrics_by_key

    def to_string(self):
        metrics, metrics_by_key = self.get_metrics()
        out = []
        out.append(tabulate(metrics.items(), headers=["Metric", "Value"], tablefmt="github"))

        for metric, values in metrics_by_key.items():
            out.append(metric + "\n" + tabulate(values.items(), tablefmt="github"))

        return "\n\n".join(out)

    def update_metrics_by_keys(self, metric: str, keys: torch.Tensor, values: torch.Tensor):
        for k, v in zip(keys, values.float()):
            self.metrics_by_key[metric][k.item()].add(v)

    def update_metrics(
        self,
        trajectories: Trajectories,
        graph: Graph,
        observations,
        observed,
        starts,
        targets,
        predictions,
        rw_weights,
        trajectory_idx,
        rw_non_backtracking,
    ):
        n_pred = len(starts)
        # remove added self loops
        rw_weights = rw_weights[:, : graph.n_edge]

        target_distributions = observations[targets]

        target_probabilities = compute_target_probability(target_distributions, predictions)
        self.metrics["target_probability"].add_all(target_probabilities)

        top1_contains_target = compute_topk_contains_target(target_distributions, predictions, k=1)
        self.metrics["precision_top1"].add_all(top1_contains_target)
        top5_contains_target = compute_topk_contains_target(target_distributions, predictions, k=5)
        self.metrics["precision_top5"].add_all(top5_contains_target)

        assert trajectories.has_traversed_edges
        noise_level = 1e-6  # very small noise is added to break the uniform cases

        # [n_pred, n_node]
        _, chosen_edge_at_each_node = scatter_max(
            rw_weights + torch.rand_like(rw_weights) * noise_level, graph.senders, fill_value=-1
        )
        if rw_non_backtracking:
            nb_rw_graph = graph.update(
                edges=rw_weights.transpose(0, 1)
            ).non_backtracking_random_walk_graph
            # [n_edge, n_pred]
            _, chosen_hyperedge_at_each_edge = scatter_max(
                nb_rw_graph.edges + torch.rand_like(nb_rw_graph.edges) * noise_level,
                nb_rw_graph.senders,
                dim=0,
                fill_value=-1000,
            )
            chosen_edge_at_each_edge = nb_rw_graph.receivers[chosen_hyperedge_at_each_edge]
            # [n_pred, n_edge]
            chosen_edge_at_each_edge = chosen_edge_at_each_edge.transpose(0, 1)

        for pred_id in range(n_pred):
            # concat all edges traversed between start and target
            traversed_edges = torch.cat(
                [
                    trajectories.traversed_edges(trajectory_idx, i)
                    for i in range(starts[pred_id], targets[pred_id])
                ]
            )
            # remove consecutive duplicate
            duplicate_mask = torch.zeros_like(traversed_edges, dtype=torch.uint8)
            duplicate_mask[1:] = traversed_edges[:-1] == traversed_edges[1:]
            traversed_edges = traversed_edges[~duplicate_mask]

            nodes_where_decide = graph.senders[traversed_edges]

            """ choice accuracy """

            if rw_non_backtracking:
                chosen_edges = torch.zeros_like(traversed_edges, dtype=torch.long)
                first_node = nodes_where_decide[0]
                chosen_edges[0] = chosen_edge_at_each_node[pred_id, first_node]
                chosen_edges[1:] = chosen_edge_at_each_edge[pred_id, traversed_edges[:-1]]
            else:
                chosen_edges = chosen_edge_at_each_node[pred_id, nodes_where_decide]

            correct_choices = (traversed_edges == chosen_edges).float()
            self.metrics["choice_accuracy"].add_all(correct_choices)

            deg3_mask = graph.out_degree_counts[nodes_where_decide] > 2
            deg3_mask[0] = 1
            self.metrics["choice_accuracy_deg3"].add_all(correct_choices[deg3_mask])

            """NLL computation"""

            if not rw_non_backtracking:
                traversed_edges_weights = rw_weights[pred_id, traversed_edges]
            else:
                rw_graph = graph.update(edges=rw_weights[pred_id])
                nb_rw_graph = rw_graph.non_backtracking_random_walk_graph

                traversed_edges_weights = torch.zeros(len(traversed_edges))
                traversed_edges_weights[0] = rw_weights[pred_id, traversed_edges[0]]
                for i, (s, r) in enumerate(zip(traversed_edges[:-1], traversed_edges[1:])):
                    traversed_edges_weights[i + 1] = nb_rw_graph.edge(s, r)

            neg_log_weights = -(traversed_edges_weights + 1e-20).log()
            self.metrics["path_nll"].add(neg_log_weights.sum().item())
            deg3_mask = graph.out_degree_counts[graph.senders[traversed_edges]] > 2
            deg3_mask[0] = 1
            self.metrics["path_nll_deg3"].add(neg_log_weights[deg3_mask].sum().item())

        if self.config.dataset == "wikispeedia":
            jump_lengths = targets - starts

            """top k by jump"""
            self.update_metrics_by_keys("precision_top1", jump_lengths, top1_contains_target)
            self.update_metrics_by_keys("precision_top5", jump_lengths, top5_contains_target)

            """cumulative reciprocal rank"""
            # assumes only one target per observations
            target_nodes = observations[targets].nonzero()[:, 1]
            target_ranks = compute_rank(predictions, target_nodes)
            self.update_metrics_by_keys(
                "target_rank", jump_lengths, self.harmonic_numbers[target_ranks - 1]
            )

            """West target accuracy"""
            start_nodes = observations[starts].nonzero()[:, 1]
            target2_acc = target2_accuracy(
                start_nodes,
                target_nodes,
                predictions,
                self.given_as_target,
                trajectories.pairwise_node_distances,
            )

            self.update_metrics_by_keys("target2_acc", jump_lengths, target2_acc)


def target2_accuracy(
    start_nodes, target_nodes, predictions, given_as_target, pairwise_node_distances
):
    """Compute accuracy of a 2 classifier, decide between the true target and a "fake" one
    sampled from same Shortest Path Distance to start and having been given as target.

    Args:
        start_nodes: [n_pred, ]
        target_nodes: [n_pred, ]
        predictions: [n_pred, n_node]
        given_as_target: [n_node]
        pairwise_node_distances: [n_node, n_node]

    Returns:
        [n_pred, ]
    """
    noise_level = 1e-6  # very small noise is added to break the uniform cases
    predictions += torch.rand_like(predictions) * noise_level

    # n_pred
    distance_start_target = pairwise_node_distances[start_nodes, target_nodes]
    # n_pred x num_nodes
    distance_from_starts_to_every_nodes = pairwise_node_distances[start_nodes, :]
    # n_pred x num_nodes
    right_distance_mask = distance_from_starts_to_every_nodes == distance_start_target.unsqueeze(1)

    # n_pred x num_nodes
    possible_targets_mask = right_distance_mask & given_as_target.unsqueeze(0)
    # n_pred
    range_ = torch.arange(start_nodes.shape[0], device=start_nodes.device)
    # n_pred
    target_predictions = predictions[range_, target_nodes]

    #  n_pred x num_nodes
    correctly_classified = predictions < target_predictions.unsqueeze(1)
    correctly_classified &= possible_targets_mask

    # n_pred
    count_possible_target_pairs = possible_targets_mask.float().sum(dim=1) - 1
    count_correct_target_pairs = correctly_classified.float().sum(dim=1)
    target2_acc = count_correct_target_pairs / count_possible_target_pairs

    return target2_acc


def compute_target_probability(target_distributions, predicted_distributions):
    """Compute predicted probability mass that reaches a non zero probability target state

    Args:
        target_distributions: [n_pred, n_node]
        predicted_distributions: [n_pred, n_node]

    Example:
        >>> import torch
        >>> compute_target_probability(torch.tensor([[.2, .8, 0.],
        ...                                          [.1, 0., .9]]),
        ...                            torch.tensor([[.6, 0., .4],
        ...                                          [.3, .1, .6]]))
        tensor([0.6000, 0.9000])

    Returns:
        [n_pred, ]
    """
    target_mask = target_distributions > 0
    return (predicted_distributions * target_mask.float()).sum(dim=1)


def compute_topk_contains_target(target_distributions, predicted_distributions, k):
    """Compute precition on top k

    Args:
        target_distributions: [n_pred, n_node]
        predicted_distributions: [n_pred, n_node]
        k (int): k top prediction to consider in predicted_distributions

    Example:
        >>> compute_topk_contains_target(
        ...     target_distributions=torch.tensor([[0.0, 1.0, 0.0], [0.1, 0.0, 0.9]]),
        ...     predicted_distributions=torch.tensor([[0.6, 0.0, 0.4], [0.3, 0.1, 0.6]]),
        ...     k=2,
        ... )
        tensor([0, 1], dtype=torch.uint8)

    Returns:
        [n_pred, ] 1 if the top 5 predictions contains elements of the target_distributions
    """
    device = predicted_distributions.device
    _, topk_nodes = torch.topk(predicted_distributions, k)
    indices = torch.arange(len(predicted_distributions), device=device, dtype=torch.long)
    indices = indices.view(-1, 1).repeat(1, k)  # [[0, ... 0], [1, ..., 1], ....]
    topk_target_values = target_distributions[indices, topk_nodes]
    return (topk_target_values > 0).any(dim=1)


def compute_rank(predictions, targets):
    """Compute the rank (between 1 and n) of of the true target in ordered predictions

    Example:
        >>> import torch
        >>> compute_rank(torch.tensor([[.1, .7, 0., 0., .2, 0., 0.],
        ...                            [.1, .7, 0., 0., .2, 0., 0.],
        ...                            [.7, .2, .1, 0., 0., 0., 0.]]),
        ...              torch.tensor([4, 1, 3]))
        tensor([2, 1, 5])

    Args:
        predictions (torch.Tensor): [n_pred, n_node]
        targets (torch.Tensor): [n_pred]
    """
    n_pred = predictions.shape[0]
    range_ = torch.arange(n_pred, device=predictions.device, dtype=torch.long)

    proba_targets = predictions[range_, targets]
    target_rank_upper = (proba_targets.unsqueeze(1) < predictions).long().sum(dim=1) + 1
    target_rank_lower = (proba_targets.unsqueeze(1) <= predictions).long().sum(dim=1)

    # break tighs evenly by taking the mean rank
    target_rank = (target_rank_upper + target_rank_lower) / 2
    return target_rank

