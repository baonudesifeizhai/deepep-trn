"""DeepEP-style expert-parallel dispatch/combine for AWS Trainium."""
from .buffer import Buffer, Config, DispatchHandle, EventOverlap, LowLatencyHandle
from .codec import RowCodec
from .layout import get_dispatch_layout

__all__ = ['Buffer', 'Config', 'DispatchHandle', 'EventOverlap', 'LowLatencyHandle', 'RowCodec',
           'get_dispatch_layout']
__version__ = '0.1.0'
