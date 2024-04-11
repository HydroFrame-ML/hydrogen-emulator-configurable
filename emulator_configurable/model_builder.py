
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


# Set the concrete instance, but make it look like a singleton
ModelBuilder = ModelFactory()
