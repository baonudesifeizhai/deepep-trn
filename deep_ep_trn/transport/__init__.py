from .base import Transport
from .gloo import GlooTransport


def NkiA2avTransport(*args, **kwargs):
    """Lazy constructor so CPU-only users never import the Neuron stack."""
    from .nki_a2av import NkiA2avTransport as cls
    return cls(*args, **kwargs)


__all__ = ['Transport', 'GlooTransport', 'NkiA2avTransport']
