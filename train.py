import os
import shutil
import random
import numpy as np
import torch
import torch.utils.data
import torch.distributed as dist
from config import config_parser
from tensorboardX import SummaryWriter
from loaders.create_training_dataset import get_training_dataset
from trainer import BaseTrainer

torch.manual_seed(1234)

# ✅ Force CPU-only training
device = torch.device("cpu")
torch.set_default_dtype(torch.float32)

def synchronize():
    """Synchronize when using distributed training (disabled for CPU-only mode)."""
    if not dist.is_available() or not dist.is_initialized():
        return
    if dist.get_world_size() == 1:
        return
    dist.barrier()

def seed_worker(worker_id):
    """Ensure dataset workers have deterministic seeds."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def train(args):
    """Train OmniMotion on the football dataset."""
    seq_name = "football_tracking"
    out_dir = os.path.join(args.save_dir, f"{args.expname}_{seq_name}")
    os.makedirs(out_dir, exist_ok=True)
    print(f"⚽ Training OmniMotion for {seq_name}...\n📁 Output will be saved in {out_dir}")

    args.out_dir = out_dir

    # ✅ Save training config
    with open(os.path.join(out_dir, "args.txt"), "w") as file:
        for arg in sorted(vars(args)):
            if not arg.startswith("_"):
                file.write(f"{arg} = {getattr(args, arg)}\n")

    if args.config:
        config_path = os.path.join(out_dir, "config.txt")
        if not os.path.isfile(config_path):
            shutil.copy(args.config, config_path)

    # ✅ Set up TensorBoard logging
    log_dir = f"logs/{args.expname}_{seq_name}"
    writer = SummaryWriter(log_dir)

    # ✅ Load dataset (football images)
    g = torch.Generator()
    g.manual_seed(args.loader_seed)
    print("📂 Loading dataset...")  # Debugging print
    dataset, data_sampler = get_training_dataset(args, max_interval=args.start_interval)
    print("✅ Dataset Loaded!")  # Debugging print

    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.num_pairs,  # Reduce if memory issues occur
        worker_init_fn=seed_worker,
        generator=g,
        num_workers=2,  # Adjust to prevent CPU overload
        sampler=data_sampler,
        shuffle=True if data_sampler is None else False,
        pin_memory=True,
    )

    # ✅ Initialize OmniMotion Trainer
    print("🚀 Initializing Trainer...")
    trainer = BaseTrainer(args)
    print("✅ Trainer Ready!")

    start_step = trainer.step + 1
    step = start_step
    epoch = 0

    while step < args.num_iters + start_step + 1:
        for batch in data_loader:
            trainer.train_one_step(step, batch)
            trainer.log(writer, step)
            step += 1
            dataset.set_max_interval(args.start_interval + step // 2000)

            if step >= args.num_iters + start_step + 1:
                break

        epoch += 1
        if args.distributed:
            data_sampler.set_epoch(epoch)

if __name__ == "__main__":
    args = config_parser()
    
	
    args.batch_size = 4  # Ensure batch size is set
    args.learning_rate = 0.0001  # Ensure learning rate is set
    print(f"🔹 Loaded Config: num_pairs={args.num_pairs}, lr_feature={args.lr_feature}, lr_deform={args.lr_deform}, lr_color={args.lr_color}")
  
    # ✅ Ensure dataset path is set to your football dataset
    args.data_dir = "/Users/nadiajelani/Documents/GitHub/omni/dataset"
    args.save_dir = "/Users/nadiajelani/Documents/GitHub/omni/outputs"
    args.expname = "football_experiment"
    args.num_iters = 50000  # Adjust training iterations
    args.num_pairs = 4  # Reduce batch size for stability

    if args.distributed:
        torch.distributed.init_process_group(backend="gloo", init_method="env://")
        synchronize()

    train(args)
