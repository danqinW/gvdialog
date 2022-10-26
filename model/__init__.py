from .GVDialog import GVDialog
from utils.registry import MODELS

def build_models(name, config):
    return MODELS[name](config)