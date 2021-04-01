# pylint: disable=abstract-method, arguments-differ, unused-wildcard-import

# Jonas Braun
# jonas.braun@epfl.ch
# 22.02.2021

# copied and modified from Semih Günel's repo https://github.com/NeLy-EPFL/Drosoph2PRegistration
import os.path, sys
import pytorch_lightning as pl
from pytorch_lightning import Trainer, seed_everything
import torch
import torch.nn.functional as F
import torch.nn as nn
from torchvision import utils
from torch.autograd import Variable
import gc
import glob

FILE_PATH = os.path.realpath(__file__)
DEEPREG_PATH, _ = os.path.split(FILE_PATH)
REGISTER_PATH, _ = os.path.split(DEEPREG_PATH)
LONGTERM_PATH, _ = os.path.split(REGISTER_PATH)
MODULE_PATH, _ = os.path.split(LONGTERM_PATH)
sys.path.append(MODULE_PATH)

from longterm.register.deepreg.model import UNetSmall, GridSpatialIntegral, Warper
from longterm.register.deepreg.dataset import Dataset2PLight
from longterm.register.deepreg.loss import TotalVaryLoss, BiasReduceLoss


def getBaseGrid(Nx, Ny=None, normalize=True, getbatch=False, batchSize=1):
    Ny = Ny if Ny is not None else Ny
    ax = torch.arange(-(Nx - 1.0), (Nx), 2)
    ay = torch.arange(-(Ny - 1.0), (Ny), 2)
    if normalize:
        ax = ax / (Nx - 1)
        ay = ay / (Ny - 1)
    x = ax.repeat(Ny, 1)
    y = ay.repeat(Nx, 1).t()
    grid = torch.cat((x.unsqueeze(0), y.unsqueeze(0)), 0)
    if getbatch:
        grid = grid.unsqueeze(0).repeat(batchSize, 1, 1, 1)
    return grid


class Register(pl.LightningModule):
    def __init__(self, image_size_x, image_size_y=None, loss_factor_smooth=100):
        super().__init__()
        # self.save_hyperparameters()
        self.image_size_x = image_size_x
        self.image_size_y = image_size_y if image_size_y is not None else image_size_x
        self.min_image_size = min([self.image_size_x, self.image_size_y])

        self.model_deform = nn.Sequential(
            UNetSmall(num_channels=2, out_channels=2), nn.Sigmoid()
        )
        self.integrator = GridSpatialIntegral(wx=self.image_size_x, wy=self.image_size_y)
        self.cutter = nn.Hardtanh(-1, 1)
        self.warper = Warper()
        self.base_grid = getBaseGrid(Nx=self.image_size_x, Ny=self.image_size_y).cuda()

        self.tv_loss = TotalVaryLoss()
        self.bias_loss = BiasReduceLoss()
        self.loss_factor_smooth = loss_factor_smooth

        self.add_module("deform", self.model_deform)

        self.zero_warp = Variable(
            torch.cuda.FloatTensor(1, 2, self.image_size_y, self.image_size_x).fill_(0)
        )

    def viz(self, x, name):
        x = x[:16]
        self.logger.experiment.add_image(
            name,
            utils.make_grid(x, nrow=4, normalize=True, scale_each=True),
            self.current_epoch,
        )

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-4)

    def forward(self, I_t, I_k):
        I_deform = self.model_deform(torch.cat([I_t, I_k], dim=1)) * (5.0 / self.min_image_size)  #3  5/image_size sets a boundary on the derivative of the deformation field

        I_deform = self.integrator(I_deform) - 1.2
        I_deform = self.cutter(I_deform)
        I_applied = self.warper(I_t, I_deform)
        return I_deform, I_applied

    def step(self, batch, batch_idx, n):
        I_t, _, I_k = batch

        I_t_deform, I_t_applied, = self(I_t, I_k)

        recons_loss = F.mse_loss(I_k, I_t_applied)
        smooth_loss = self.tv_loss(I_t_deform)
        ident_loss = self.bias_loss(I_t_deform - self.base_grid, self.zero_warp)

        loss = self.loss_factor_smooth*smooth_loss + ident_loss + recons_loss

        if (batch_idx % 100) == 0:
            self.viz(
                torch.cat([I_t, I_k, I_t_applied, I_k - I_t_applied], dim=3),
                f"{n}/recons",
            )

        self.log(f"{n}/loss", loss)
        # self.log(f"{n}/loss_content", content_loss)
        self.log(f"{n}/loss_smooth", smooth_loss)
        self.log(f"{n}/loss_recons", recons_loss)
        self.log(f"{n}/loss_ident", ident_loss)

        return loss

    def training_step(self, batch, batch_idx):
        return self.step(batch, batch_idx, n="train")


if __name__ == "__main__":
    seed_everything(123)
    torch.cuda.empty_cache()
    gc.collect()

    datasize = "128"  # "128"  # "256"  # "480"  # "full"  # "continue_full"
    loss_factor_smooth = 10
    batch_size = 32

    if datasize == "128":
        contrastive = Register(image_size_x=128, image_size_y=128, loss_factor_smooth=loss_factor_smooth).float()
        dataset = Dataset2PLight(path="/home/jbraun/data/longterm/210212/Fly1/processed/reg_train_data_128.npy",
                                 batch_size=batch_size)

    if datasize == "256":
        contrastive = Register(image_size_x=256, image_size_y=256, loss_factor_smooth=loss_factor_smooth).float()
        dataset = Dataset2PLight(path="/home/jbraun/data/longterm/210212/Fly1/processed/reg_train_data_256.npy",
                                 batch_size=batch_size)
        
    elif datasize == "max":
        print("Using image size max")
        contrastive = Register(image_size_x=422, image_size_y=422, loss_factor_smooth=loss_factor_smooth).float()
        dataset = Dataset2PLight(path="/home/jbraun/data/longterm/210212/Fly1/processed/reg_train_data_max.npy",
                                 batch_size=batch_size)

    elif datasize == "full":
        print("Using image size full")
        contrastive = Register(image_size_x=640, image_size_y=416, loss_factor_smooth=loss_factor_smooth).float()
        dataset = Dataset2PLight(path="/home/jbraun/data/longterm/210212/Fly1/processed/reg_train_data_640_416.npy",
                                 batch_size=batch_size)

    elif datasize == "continue_full":
        ckpt_path = glob.glob(os.path.join("/home/jbraun/data/longterm/210212/Fly1/processed/lightning_logs/version_0", "checkpoints/") + "epoch*.ckpt")[0]
        print(ckpt_path)

        # contrastive = Register(image_size_x=640, image_size_y=416, loss_factor_smooth=loss_factor_smooth).float()
        contrastive = Register.load_from_checkpoint(checkpoint_path=ckpt_path, 
                                                    image_size_x=640, image_size_y=416, 
                                                    loss_factor_smooth=loss_factor_smooth).eval().cuda()
        contrastive.eval()

        dataset = Dataset2PLight(path="/home/jbraun/data/longterm/210212/Fly1/processed/reg_train_data_640_416.npy",
                                 batch_size=batch_size)

    trainer = Trainer(gpus=1, default_root_dir="/home/jbraun/data/longterm/210212/Fly1/processed")
    trainer.fit(contrastive, dataset)
