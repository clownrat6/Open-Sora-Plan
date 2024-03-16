from typing import Union, Tuple

from ..vqvae import VQVAEConfiguration


class VQVAEIUConfiguration(VQVAEConfiguration):
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
