import torch
import os

from torch.nn import functional as F
from emulator_configurable import utils

def register_layer(key):
    """
    The layer decorator is used to register a layer for the ModelBuilder.
    Layers are used to define the structure of the neural network model.
    They do not constitute a complete model, but rather a building block
    that can be used to construct a model.
    """
    def decorator(layer):
         ModelBuilder.registry['layer'][key] = layer
         return layer
    return decorator


def register_model(key):
    """
    The model decorator is used to register a model for the ModelBuilder.
    Models are complete neural network architectures that can be used
    for generic purposes, not necessarily tied to the parflow emulation.
    """
    def decorator(model):
        ModelBuilder.registry['model'][key] = model
        return model
    return decorator


def register_emulator(key):
    """
    The emulator decorator is used to register a model for the ModelBuilder.
    Emulators are complete neural network architectures that are specifically
    designed to emulate the ParFlow model.
    """
    def decorator(emulator):
        ModelBuilder.registry['emulator'][key] = emulator
        return emulator
    return decorator


class ModelFactory:
    """
    The model factory is used to generate concrete implementations
    of classes defined in the models module from configuration files.
    """

    def __init__(self):
        super().__init__()
        self.registry = {
            'layer': {},
            'model': {},
            'emulator': {},
        }

    def __repr__(self):
        message = str(self.registry)
        return message

    def build_emulator(self, type, config):
        layer_model = config.get('layer_model', None)
        if layer_model:
            config['layer_model'] = self.registry['model'][layer_model]
        ModelClass = self.registry['emulator'][type]
        return ModelClass(**config)


def model_setup(
    model_type,
    model_config,
    learning_rate=None,
    gradient_loss_penalty=True,
    model_weights=None,
    precision=torch.float32,
    device='cuda',
):
    """
    Set up the model for training or inference.

    Args:
        model_type (str): The type of model to build.
        model_config (dict): The configuration parameters for building the model.
        learning_rate (float, optional): The learning rate for the optimizer. Defaults to None.
        gradient_loss_penalty (bool, optional): Whether to use the spatial gradient penalty loss. Defaults to True.
        model_weights (dict, optional): The weights of the model saved during training. Defaults to None.
        precision (torch.dtype, optional): The precision of the model. Defaults to torch.float32.
        device (str, optional): The device to use for training or inference. Defaults to 'cuda'.

    Returns:
        Model: The configured model.
    """
    # Create the model structure
    model = ModelBuilder.build_emulator(type=model_type, config=model_config)
    # Load the state dictionary if it is provided
    # These are the weights of the model that were saved during training
    if model_weights:
        model.load_state_dict(model_weights)
    # Move the model to the device and precision specified
    if device:
        model.to(device)
    if precision:
        model.to(precision)

    # Configure the loss function. If gradient_loss_penalty is True, use the
    # spatial gradient penalty loss, otherwise use the default mse_loss.
    # The spatial gradient penalty loss adds an additional term to the 
    # loss which accounts for the spatial gradient of the output, calculated
    # via a simple finite difference.
    if gradient_loss_penalty:
        model.configure_loss(loss_fun=utils.spatial_gradient_penalty_loss)
    else:
        model.configure_loss(loss_fun=F.mse_loss)
    if learning_rate:
        model.learning_rate = learning_rate
    return model


# Set the concrete instance, but make it look like a singleton
ModelBuilder = ModelFactory()