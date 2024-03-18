import os
import math
import argparse
from time import time
from typing import Optional

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import DataLoader

from accelerate import Accelerator
from accelerate.utils import set_seed
from dataclasses import dataclass, field, asdict
from diffusers.optimization import get_scheduler
from transformers import HfArgumentParser, TrainingArguments

from opensora.models.ae.videobase import (
    VQVAEModel, VQVAEConfiguration, 
    VQVAEIUModel, VQVAEIUConfiguration,
    VideoAETrainer, build_videoae_dataset
)
from opensora.utils.utils import get_experiment_dir, create_logger, requires_grad, update_ema, write_tensorboard, \
    cleanup, create_tensorboard

# the first flag below was False when we tested this script but True makes A100 training a lot faster:
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


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
    num_train_steps: Optional[int] = field(
        default=None, metadata={"help": "number of training steps"}
    )
    resume: Optional[str] = field(
        default=None, metadata={"help": "resume training from a checkpoint"}
    )


def main(args, ae_args, training_args):
    assert torch.cuda.is_available(), "Training currently requires at least one GPU."

    args.mixed_precision = "bf16" if args.bf16 else "fp16" if args.fp16 else "fp32"

    # Setup accelerator:
    accelerator = Accelerator(mixed_precision=args.mixed_precision, gradient_accumulation_steps=args.gradient_accumulation_steps)
    device = accelerator.device
    world_size = accelerator.num_processes

    # Setup logging and tensorboard:
    checkpoint_dir = os.path.join(args.output_dir, 'checkpoints')
    experiment_dir = os.path.join(args.output_dir, 'experiments')
    if accelerator.is_main_process:
        os.makedirs(checkpoint_dir, exist_ok=True)  # Make results folder (holds all experiment subfolders)
        os.makedirs(experiment_dir, exist_ok=True)
        logger = create_logger(args.output_dir)
        tb_writer = create_tensorboard(experiment_dir)
        logger.info(f"Experiment directory created at {experiment_dir}")
        logger.info(f'{args}')
    else:
        logger = create_logger(None)
        tb_writer = None

    # Load Config
    config = VQVAEConfiguration(**asdict(ae_args))
    if training_args.train_model == "vqvae":
        model = VQVAEModel(config)
    elif training_args.train_model == "vqvaeiu":
        model = VQVAEIUModel(config)

    # Setup optimizer
    opt = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    # Load Dataset
    dataset = build_videoae_dataset(args.data_path, sequence_length=args.sequence_length, resolution=config.resolution)

    # Setup dataloader:
    loader = DataLoader(
        dataset,
        batch_size=int(args.per_device_train_batch_size),
        shuffle=False,
        # sampler=sampler,
        num_workers=args.dataloader_num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=True,
    )
    logger.info(f"Dataset contains {len(dataset):,} videos ({args.data_path})")

    # Setup learning rate scheduler:
    if args.num_train_steps is None:
        num_train_epochs = int(args.num_train_epochs)
        num_train_steps = math.ceil(num_train_epochs * len(loader) / world_size)
        lr_warmup_steps = math.ceil(args.warmup_ratio * num_train_steps)
    else:
        num_train_steps = args.num_train_steps
        lr_warmup_steps = math.ceil(args.warmup_ratio * num_train_steps)
        num_train_epochs = math.ceil(num_train_steps / len(loader) * world_size)

    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=opt,
        num_warmup_steps=lr_warmup_steps * world_size // args.gradient_accumulation_steps,
        num_training_steps=num_train_steps * world_size // args.gradient_accumulation_steps,
    )

    # TODO: Add support for resuming training from a checkpoint
    first_step = 1
    if training_args.resume:
        checkpoint = torch.load(training_args.resume, map_location='cpu')
        model.load_state_dict(checkpoint["model"])
        opt.load_state_dict(checkpoint["opt"])
        args = checkpoint["args"]
        first_step = checkpoint["step"] + 1
        if accelerator.is_main_process:
            logger.info(f"Resuming training from {training_args.resume}")

    model.train()  # important! This enables embedding dropout for classifier-free guidance
    wrapped_model, opt, loader, lr_scheduler = accelerator.prepare(model, opt, loader, lr_scheduler)

    if accelerator.is_main_process:
        logger.info(f"Training for {args.num_train_epochs} epochs...")

    if args.mixed_precision == "bf16":
        dtype = torch.bfloat16
    elif args.mixed_precision == "fp16":
        dtype = torch.float16
    else:
        dtype = torch.float32

    # Variables for monitoring/logging purposes:
    running_loss = 0
    running_sped = []
    start_time = init_time = time()

    iter_loader = iter(loader)
    for global_step in range(1, num_train_steps + 1):
        if global_step < first_step:
            continue
        data = next(iter_loader)

        x = data['video'].to(device, non_blocking=True)

        with accelerator.accumulate(wrapped_model):
            with torch.autocast(device_type='cuda', dtype=dtype):
                loss = wrapped_model(x)

            accelerator.backward(loss)
            gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            opt.step()
            lr_scheduler.step()
            opt.zero_grad()

        # Log loss values:
        running_loss += loss.item()
        if global_step % args.logging_steps == 0:
            cur_lr = lr_scheduler.get_last_lr()[0]
            log_steps = args.logging_steps
            # calculate epoch:
            epoch = global_step // len(loader) + 1
            # Measure training speed:
            torch.cuda.synchronize()
            end_time = time()
            secs_per_step = (end_time - start_time) / log_steps
            # Reduce loss history over all processes:
            avg_loss = torch.tensor(running_loss / log_steps, device=device)
            dist.all_reduce(avg_loss, op=dist.ReduceOp.SUM)
            avg_loss = avg_loss.item() / world_size
            # Calculate remaining time:
            running_sped.append(secs_per_step)
            running_sped = running_sped[-10:]
            stable_sped = torch.tensor(running_sped[-10:]).median().item()
            sec_to_hms = lambda x: "{:02d}:{:02d}:{:02d}".format(int(x // 3600), int(x % 3600 // 60), int(x % 60))
            remaining_time = (num_train_steps - global_step) * stable_sped
            remaining_time = sec_to_hms(remaining_time)
            running_time = sec_to_hms(end_time - init_time)
            if accelerator.is_main_process:
                logger.info(
                    f"(Epoch: {epoch:03d}/{num_train_epochs:03d} Step: {global_step:07d}/{num_train_steps:07d} {running_time}<{remaining_time}) " + 
                    f"Train Loss: {avg_loss:.4f}, Learning Rate: {cur_lr:.6f}"
                )

            write_tensorboard(tb_writer, 'Train Loss', avg_loss, global_step)
            write_tensorboard(tb_writer, 'Learning Rate', cur_lr, global_step)
            write_tensorboard(tb_writer, 'Train Secs/Step', secs_per_step, global_step)

            # Reset monitoring variables:
            running_loss = 0

            start_time = time()

        if accelerator.is_main_process:
            # Save Model checkpoint:
            if global_step % args.save_steps == 0:
                checkpoint = {
                    "model": model.state_dict(),
                    "opt": opt.state_dict(),
                    "step": global_step,
                    "args": args
                }
                checkpoint_path = f"{checkpoint_dir}/{global_step:07d}"
                torch.save(checkpoint, checkpoint_path + '.pt')
                model.save_checkpoint(checkpoint_path)
                logger.info(f"Saved checkpoint to {checkpoint_path}")
            elif global_step == num_train_steps:
                checkpoint = {
                    "model": model.state_dict(),
                    "opt": opt.state_dict(),
                    "step": global_step,
                    "args": args
                }
                checkpoint_path = f"{checkpoint_dir}/final"
                torch.save(checkpoint, checkpoint_path + '.pt')
                model.save_checkpoint(checkpoint_path)
                logger.info(f"Saved checkpoint to {checkpoint_path}")

    model.eval()  # important! This disables randomized embedding dropout
    # do any sampling/FID calculation/etc. with ema (or model) in eval mode ...

    if accelerator.is_main_process:
        logger.info("Done!")
    cleanup()


if __name__ == "__main__":
    parser = HfArgumentParser((AEArgument, AETrainingArgument))
    ae_args, training_args = parser.parse_args_into_dataclasses()
    args = argparse.Namespace(**vars(ae_args), **vars(training_args))
    set_seed(args.seed)

    main(args, ae_args, training_args)
