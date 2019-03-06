import torch


def any_nan(tensor: torch.Tensor) -> bool:
    """Returns true if the tensor contains a NaN

    Args:
        tensor (torch.Tensor): the input tensor

    Returns:
        bool: true if contains a NaN
    """
    return bool(torch.isnan(tensor).any().item())


def print_min_max(name, tensor):
    """Print information about a tensor

    Args:
            name (str): tensor name
            tensor (torch.Tensor): the tensor
    """
    print(
        f"{name} | min {tensor.min()} | max {tensor.max()} | hasnan {any_nan(tensor)} | shape {tensor.shape}"
    )


def assert_allclose(tensor, value, tol=1e-5, message=""):
    """Check that all values in the tensor are close to value

    Args:
        tensor (torch.Tensor): the tensor
        value: target value(s)
        tol (float, optional): Defaults to 1e-5. tolerance
        message (str, optional): Defaults to "". displayed error message
    """
    assert ((tensor - value).abs() < tol).all(), message


def assert_proba_distribution(probabilities, tol=1e-5):
    """Check that the tensor is a probability distribution

    Args:
        probabilities (torch.Tensor): the distribution
        tol (float, optional): Defaults to 1e-5. tolerance
    """

    assert (probabilities.sum() - 1.0).abs() < tol and (
        probabilities >= 0
    ).all(), "tensor was expected to be a proability distribution (sum={}, negatives={})".format(
        probabilities.sum(), (probabilities < 0).any()
    )
