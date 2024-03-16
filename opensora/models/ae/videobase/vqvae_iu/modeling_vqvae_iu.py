import os
import math
import json
from typing import Tuple, Dict, Union

import torch
import torch.nn as nn
import torch.distributed as dist
import torch.nn.functional as F
import numpy as np
from torch import Tensor

from .configuration_vqvae_iu import VQVAEIUConfiguration
from ..modeling_videobase import VideoBaseAE
from ..vqvae.modeling_vqvae import (
    AttentionResidualBlock, Encoder, Codebook,
    SamePadConv3d, SamePadConvTranspose3d, shift_dim
)


# Modified from https://github.com/wilson1yan/VideoGPT
class Decoder(nn.Module):
    def __init__(self, n_hiddens, n_res_layers, upsample):
        super().__init__()
        self.res_stack = nn.Sequential(
            *[AttentionResidualBlock(n_hiddens) for _ in range(n_res_layers)],
            nn.BatchNorm3d(n_hiddens),
            nn.ReLU(),
        )

        n_times_upsample = np.array([int(math.log2(d)) for d in upsample])
        max_us = n_times_upsample.max()
        self.convus = nn.ModuleList()
        for i in range(max_us):
            out_channels = 3 if i == max_us - 1 else n_hiddens
            us = tuple([2 if d > 0 else 1 for d in n_times_upsample])
            if i < max_us - 1:
                convu = nn.Sequential(SamePadConv3d(n_hiddens, out_channels, 3), nn.BatchNorm3d(out_channels), nn.ReLU(inplace=True), nn.Upsample(scale_factor=us, mode='trilinear'))
            else:
                convu = nn.Sequential(SamePadConv3d(n_hiddens, out_channels, 3), nn.Upsample(scale_factor=us, mode='trilinear'), nn.Tanh())
            self.convus.append(convu)
            n_times_upsample -= 1

    def forward(self, x):
        h = self.res_stack(x)
        for i, convu in enumerate(self.convus):
            h = convu(h)
        return h


# Modified from https://github.com/wilson1yan/VideoGPT
class VQVAEIUModel(VideoBaseAE):

    def __init__(self, config: VQVAEIUConfiguration):
        super().__init__()
        self.config = config
        self.embedding_dim = config.embedding_dim
        self.n_codes = config.n_codes
        self.encoder = Encoder(config.n_hiddens, config.n_res_layers, config.downsample)
        self.decoder = Decoder(config.n_hiddens, config.n_res_layers, config.downsample)
        self.pre_vq_conv = SamePadConv3d(config.n_hiddens, config.embedding_dim, 1)
        self.post_vq_conv = SamePadConv3d(config.embedding_dim, config.n_hiddens, 1)
        self.codebook = Codebook(config.n_codes, config.embedding_dim)

    def forward(self, x):
        z = self.pre_vq_conv(self.encoder(x))
        vq_output = self.codebook(z)
        comme_loss = vq_output['commitment_loss']
        x_recon = self.decoder(self.post_vq_conv(vq_output["embeddings"]))
        recon_loss = F.mse_loss(x_recon, x) / 0.06
        if self.training:
            loss = comme_loss + recon_loss
            return loss
        else:
            return x_recon

    def encode(self, x: Tensor, include_embeddings: bool = False) -> Union[Tuple[Tensor, Tensor], Tensor]:
        h = self.pre_vq_conv(self.encoder(x))
        vq_output: Dict[str, Tensor] = self.codebook(h)
        if include_embeddings:
            return vq_output["encodings"], vq_output["embeddings"]
        else:
            return vq_output["encodings"]

    def decode(self, encodings: Tensor) -> Tensor:
        h = F.embedding(encodings, self.codebook.embeddings)
        h = self.post_vq_conv(shift_dim(h, -1, 1))
        return self.decoder(h)

    @classmethod
    def load_from_checkpoint(cls, model_path):
        if not os.path.isdir(model_path):
            """model downloaded from internet"""
            model_cpkt = torch.load(model_path)
            # Compatible with old videogpt model formats.
            if "hyper_parameters" in model_cpkt:
                hyper_parameters = vars(model_cpkt.get("hyper_parameters").get("args"))
                state_dict = model_cpkt.get("state_dict")
                model = cls(config=VQVAEConfiguration(**hyper_parameters))
                model.load_state_dict(state_dict)
                return model
            else:
                raise RuntimeError("Model checkpoint has a wrong format.")
        else:
            with open(os.path.join(model_path, "config.json"), "r") as file:
                config = json.load(file)
            state_dict = torch.load(os.path.join(model_path, "pytorch_model.bin"), map_location="cpu")
            model = cls(config=VQVAEIUConfiguration(**config))
            model.load_state_dict(state_dict, strict=False)
            return model

    @classmethod
    def download_and_load_model(cls, model_name, cache_dir=None):
        from .....utils.downloader import gdown_download
        path = gdown_download(
            cls.DOWNLOADED_VQVAE[model_name], model_name, cache_dir=cache_dir
        )
        return cls.load_from_checkpoint(path)
