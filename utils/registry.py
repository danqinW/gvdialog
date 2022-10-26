MODELS = {}

def register(name):
    global MODELS
    def outer(cls):
        MODELS[name] = cls
        return cls
    return outer