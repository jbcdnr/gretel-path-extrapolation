import configparser
import copy
import logging
from datetime import datetime
from typing import Optional
import typing
from typing import List

import torch

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Config(object):
    """Configuration object for the experiments"""

    name: str = None
    date: str = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    workspace: str = "../workspace"
    dataset: str = "mesh"
    input_directory: str = ""
    max_trajectory_length: int = 120
    min_trajectory_length: int = 6
    k_closest_nodes: int = 5
    extract_coord_features: bool = True

    device: torch.device = torch.device("cpu")
    optimizer: str = "Adam"
    loss: str = "nll_loss"
    lr: float = 0.01
    momentum: float = 0.5
    batch_size: int = 5
    overfit1: bool = False
    shuffle_samples: bool = True
    seed: int = 0
    number_epoch: int = 10
    train_test_ratio: str = "0.8/0.2"
    patience: int = 1000

    number_observations: int = 5
    self_loop_weight: float = 0.01
    self_loop_deadend_only: bool = True

    diffusion_k_hops: int = 60
    diffusion_hidden_dimension: int = 1
    parametrized_diffusion: bool = False
    target_prediction: str = "next"
    latent_transformer_see_target: bool = False

    rw_max_steps: int = -1
    rw_edge_weight_see_number_step: bool = False
    rw_expected_steps: bool = True
    with_interpolation: bool = False
    initial_edge_transformer: bool = False
    use_shortest_path_distance: bool = False
    double_way_diffusion: bool = False
    rw_non_backtracking: bool = True
    diffusion_self_loops: bool = False

    print_per_epoch: int = 10

    checkpoint_directory: str = "chkpt"
    enable_checkpointing: bool = True
    chechpoint_every_num_epoch: int = 5
    restore_from_checkpoint: bool = False
    compute_baseline: bool = True

    _DEPRECATED: List[str] = [
        "tensorboard_logdir",
        "log_tensorboard",
        "rw_restart_coef",
        "rw_tolerance",
        "rw_walk_or_die",
        "print_every",
    ]

    tensorboard_logdir: str = "logdir"
    log_tensorboard: bool = True
    rw_restart_coef: float = 0.0
    rw_tolerance: float = 0.0
    rw_walk_or_die: bool = True
    print_every: int = 10

    def load_from_file(self, filename: str):
        """Load configuration fiels from a file

        Args:
            filename (str): file name
        """

        config = configparser.ConfigParser()
        config.read(filename)
        fields = {
            key: value for section in config.sections() for key, value in config[section].items()
        }
        self.load_from_dict(fields)

    def load_from_dict(self, fields: dict):
        """Load all fields from the dictionary

        Args:
            fields (dict): configuration (key, value)'s

        Raises:
            NotImplementedError: Unknown configuration keys
        """

        for key, value in fields.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise NotImplementedError(f'Unknown configuration field "{key}"')

    def __getattribute__(self, k):
        deprecated = super(Config, self).__getattribute__("_DEPRECATED")
        if k in deprecated:
            logger.warning(f"Accessed to deprecated config field '{k}'")
        return super(Config, self).__getattribute__(k)

    def __setattr__(self, k, v):
        """Update a configuration key, take care of casting

        Args:
            k (str): configuration
            v: value

        Raises:
            NotImplementedError: Unkown type of field for casting
            AttributeError: Unknown attribute
        """

        type_annotations = typing.get_type_hints(self)
        if k in type_annotations:
            typ = type_annotations[k]
            if type(v) is not typ:
                if typ is int:
                    v = int(v)
                elif typ is float:
                    v = float(v)
                elif typ is bool:
                    v = v.lower() in ["yes", "true", "1"]
                elif typ is torch.device:
                    v = torch.device(v)
                else:
                    raise NotImplementedError(f"Unknown config type '{typ}' for field '{k}'")
            self.__dict__[k] = v
        else:
            raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{k}'")

    def to_dict(self):
        return {k: v for k, v in self.__dict__.items() if k[0] != "_" and k not in self._DEPRECATED}

    def save_to_file(self, filename):
        """Save this configuration object to a file

        Args:
            filename (str): path of the file
        """
        with open(filename, "w") as f:
            f.write("[config]\n")
            for k, v in self.to_dict().items():
                f.write(f"{k} = {v}\n")

    def __str__(self):
        type_annotations = typing.get_type_hints(self)
        lines = ["[config]"] + [f"{k} = {getattr(self, k)}" for k in type_annotations.keys()]
        return "\n".join(lines)


def config_generator(config: Config, parameters: list, selected_params=None):
    """Generate alternative Config objects for grid search

    Args:
        config (Config): the initial configuration
        parameters (list): [('param_name_1', [values...]), ... ('param_name_n', [values...])]
        selected_params ([type], optional): Defaults to None.
    """

    selected_params = selected_params or []

    if parameters:
        curr_param, values = parameters[-1]
        for v in values:
            new_config = copy.copy(config)
            setattr(new_config, curr_param, v)
            new_selected_params = (selected_params or []) + [(curr_param, v)]
            for conf in config_generator(new_config, parameters[:-1], new_selected_params):
                yield conf
    else:
        config.name += "-".join("{}:{}".format(k, v) for k, v in selected_params)
        yield config
