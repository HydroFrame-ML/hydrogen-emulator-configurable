
def register_layer(key):
    def decorator(layer):
         ModelBuilder.registry['layer'][key] = layer
         return layer
    return decorator


def register_model(key):
    def decorator(model):
        ModelBuilder.registry['model'][key] = model
        return model
    return decorator


def register_emulator(key):
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

    def build_emulator(self, type, model_config):
        layer_model = model_config.pop('layer_model', None)
        if layer_model:
            model_config['layer_model'] = self.registry['model'][layer_model]
        ModelClass = self.registry['emulator'][type]
        return ModelClass(**model_config)


ModelBuilder = ModelFactory()
