from .standard_activations import ReLU, LeakyReLU, ELU, GELU, Mish, Swish
from .custom_activations import RoCLU


def get_activation(activation_name, **kwargs):
    """Return an activation module by name."""
    activations = {
        'relu': ReLU,
        'leaky_relu': LeakyReLU,
        'elu': ELU,
        'gelu': GELU,
        'mish': Mish,
        'swish': Swish,
        'roclu': RoCLU,
    }

    activation_name = activation_name.lower()
    if activation_name not in activations:
        raise ValueError(
            f"Activation '{activation_name}' not supported. "
            f"Available activations: {list(activations.keys())}"
        )

    return activations[activation_name](**kwargs)


AVAILABLE_ACTIVATIONS = [
    'relu', 'leaky_relu', 'elu', 'gelu', 'mish', 'swish', 'roclu'
]

__all__ = [
    'ReLU', 'LeakyReLU', 'ELU', 'GELU', 'Mish', 'Swish', 'RoCLU',
    'get_activation', 'AVAILABLE_ACTIVATIONS'
]
