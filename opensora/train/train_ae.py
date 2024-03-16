import argparse
from typing import Optional

from accelerate.utils import set_seed
from transformers import HfArgumentParser, TrainingArguments
from dataclasses import dataclass, field, asdict

import sys
sys.path.append(".")
from opensora.models.ae.videobase import (
    VQVAEModel, VQVAEConfiguration, 
    VQVAEIUModel, VQVAEIUConfiguration,
    VideoAETrainer, build_videoae_dataset
)

@dataclass
class AEArgument:
    embedding_dim: int = field(default=256),
    n_codes: int = field(default=2048),
    n_hiddens: int = field(default=240),
    n_res_layers: int = field(default=4),
    resolution: int = field(default=128),
    sequence_length: int = field(default=16),
    downsample: str = field(default="4,4,4"),
    no_pos_embd: bool = True,
    data_path: str = field(default=None, metadata={"help": "data path"})


@dataclass
class AETrainingArgument(TrainingArguments):
    remove_unused_columns: Optional[bool] = field(
        default=False, metadata={"help": "Remove columns not required by the model when using an nlp.Dataset."}
    )
    train_model: Optional[str] = field(
        default="vqvae", metadata={"help": "train model"}
    )


def train(args, ae_args, training_args):
    # Load Config
    config = VQVAEConfiguration(**asdict(ae_args))
    if training_args.train_model == "vqvae":
        model = VQVAEModel(config)
    elif training_args.train_model == "vqvaeiu":
        model = VQVAEIUModel(config)
    # Load Dataset
    dataset = build_videoae_dataset(args.data_path, sequence_length=args.sequence_length, resolution=config.resolution)
    # Load Trainer
    trainer = VideoAETrainer(model, training_args, train_dataset=dataset)
    trainer.train()


if __name__ == "__main__":
    parser = HfArgumentParser((AEArgument, AETrainingArgument))
    ae_args, training_args = parser.parse_args_into_dataclasses()
    args = argparse.Namespace(**vars(ae_args), **vars(training_args))
    set_seed(args.seed)

    train(args, ae_args, training_args)
