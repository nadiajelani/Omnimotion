import torch
import os
import imageio
import numpy as np
from networks.mfn import GaborNet
from networks.nvp_simplified import NVPSimplified
from kornia import morphology as morph
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(1234)
device="cpu"
class BaseTrainer:
    def __init__(self, args, device='cpu'):
        self.args = args
        self.device = torch.device("cpu")
        self.step = 0

        self.read_data()

        self.feature_mlp = GaborNet(in_size=1, hidden_size=256, n_layers=2, alpha=4.5, out_size=128).to(self.device)
        self.deform_mlp = NVPSimplified(n_layers=6, feature_dims=128, hidden_size=[256, 256, 256], proj_dims=256, 
                                        proj_type='fixed_positional_encoding', pe_freq=args.pe_freq, affine=args.use_affine,
                                        device=self.device).to(self.device)
        self.color_mlp = GaborNet(in_size=3, hidden_size=512, n_layers=3, alpha=3, out_size=4).to(self.device)
        self.optimizer = torch.optim.Adam([
            {'params': self.feature_mlp.parameters(), 'lr': args.lr_feature},
            {'params': self.deform_mlp.parameters(), 'lr': args.lr_deform},
            {'params': self.color_mlp.parameters(), 'lr': args.lr_color},
        ])

        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=args.lrate_decay_steps, gamma=args.lrate_decay_factor)
    def read_data(self):
        img_dir = self.args.data_dir
        img_files = []
        for subdir in sorted(os.listdir(img_dir)):  # ✅ Loop through train_001, train_002, etc.
            subdir_path = os.path.join(img_dir, subdir)
            if os.path.isdir(subdir_path):  # ✅ Ensure it's a directory
                img_files.extend([os.path.join(subdir_path, f) for f in sorted(os.listdir(subdir_path)) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))])

# ✅ Debugging: Print number of images found
        print(f"🔍 Found {len(img_files)} images in {img_dir}")
        self.num_imgs = min(self.args.num_imgs, len(img_files))
        self.img_files = img_files[:self.num_imgs]
    # ✅ Ignore non-image files like .DS_Store
        self.img_files = [f for f in self.img_files if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
    # ✅ Only load valid image files
        images = np.array([imageio.imread(os.path.join(img_dir, f)) / 255. for f in self.img_files])
        self.images = torch.from_numpy(images).float()  # [n_imgs, h, w, 3]
        self.h, self.w = self.images.shape[1:3]
    def train_one_step(self, step, batch):
        self.optimizer.zero_grad()
        loss = torch.randn(1, requires_grad=True)  # Dummy loss
        loss.backward()
        self.optimizer.step()
