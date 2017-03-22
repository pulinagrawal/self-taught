from functools import wraps


def ready_generator(gen):
    """
    Decorator: gets a generator gen ready
    by advancing to first yield statement
    """
    @wraps(gen)
    def generator(*args,**kwargs):
        g = gen(*args,**kwargs)
        next(g)
        return g.send
    return generator
