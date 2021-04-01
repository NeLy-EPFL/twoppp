# Jonas Braun
# jonas.braun@epfl.ch
# 23.02.2021

# copied and modified from Semih Günel's repo https://github.com/NeLy-EPFL/Drosoph2PRegistration

import os.path, sys
import glob
import numpy as np
from copy import deepcopy

import torch
from torch.nn import UpsamplingBilinear2d
from pytorch_lightning import seed_everything

import utils2p

FILE_PATH = os.path.realpath(__file__)
DEEPREG_PATH, _ = os.path.split(FILE_PATH)
REGISTER_PATH, _ = os.path.split(DEEPREG_PATH)
LONGTERM_PATH, _ = os.path.split(REGISTER_PATH)
MODULE_PATH, _ = os.path.split(LONGTERM_PATH)
sys.path.append(MODULE_PATH)

from longterm.register.deepreg.model import UNetSmall, GridSpatialIntegral, Warper
from longterm.register.deepreg.dataset import Dataset2PLight
from longterm.register.deepreg.loss import TotalVaryLoss, BiasReduceLoss
from longterm.register.deepreg.train import Register
from longterm.register import deepreg_prepare
from longterm.register import warping
from longterm import load
from longterm.utils import torch_to_numpy, get_stack
from longterm.plot.videos import make_video_2p, make_video_motion_field

class RegisterTest(Register):
    def __init__(self, image_size_x, image_size_y=None, 
                 orig_image_size_x=None, orig_image_size_y=None, loss_factor_smooth=100):
        super().__init__(image_size_x, image_size_y, loss_factor_smooth)
        self.orig_image_size_x = orig_image_size_x if orig_image_size_x is not None else image_size_x
        self.orig_image_size_y = orig_image_size_y if orig_image_size_y is not None else image_size_y
        self.integrator_origsize = GridSpatialIntegral(wx=self.orig_image_size_x, wy=self.orig_image_size_y)
        self.upsampler = UpsamplingBilinear2d(scale_factor=(self.orig_image_size_y/self.image_size_y, 
                                                            self.orig_image_size_x/self.image_size_x))
    def get_upsampled_grid(self, I_t, I_k):
        I_deform = self.model_deform(torch.cat([I_t, I_k], dim=1)) * (5.0 / self.min_image_size)  # 5/image_size sets a boundary on the derivative of the deformation field
        I_deform = self.upsampler(I_deform).cuda()
        I_deform = self.integrator_origsize(I_deform) - 1.2
        I_deform = self.cutter(I_deform)
        return I_deform


def load_model(lightninglog_basedir, image_size_x, image_size_y,
               orig_image_size_x, orig_image_size_y, loss_factor_smooth):
    ckpt_path = glob.glob(os.path.join(lightninglog_basedir, "checkpoints/") + "epoch*.ckpt")[0]
    print(ckpt_path)
    reg = RegisterTest.load_from_checkpoint(checkpoint_path=ckpt_path, 
                                        image_size_x=image_size_x, image_size_y=image_size_y, 
                                        orig_image_size_x=orig_image_size_x, orig_image_size_y=orig_image_size_y,
                                        loss_factor_smooth=loss_factor_smooth).eval().cuda()
    reg.eval()
    return reg

def apply_model_to_trial(model, red, offset_dir, original_shape=True, upsample_derivative=False, green=None, template_frame=None, crop_size=(0,0), batch_size=32,
                         red_out_dir=None, green_out_dir=None, motion_field_out_dir=None):
    model_size_x = model.image_size_x
    model_size_y = model.image_size_y

    # load stacks
    red = get_stack(red)
    green = get_stack(green) if green is not None else None

    # apply com offsets
    offset = np.load(offset_dir)
    red = warping.apply_offset(red, offset)
    green = warping.apply_offset(green, offset) if green is not None else None

    # crop
    red = deepreg_prepare.crop_stack(red, crop_size)
    green = deepreg_prepare.crop_stack(green, crop_size) if green is not None else None

    N_frames, N_y, N_x = red.shape

    # resize red channel to give to model
    if N_y != model_size_y or N_x != model_size_x:
        red_resize = deepreg_prepare.resize_stack(red, size=(model_size_y, model_size_x))
        if original_shape is False:
            red = deepcopy(red_resize)
            green = deepreg_prepare.resize_stack(green, size=(model_size_y, model_size_x))
            N_frames, N_y, N_x = red.shape
    else:
        red_resize = deepcopy(red)

    # z-score red channel to be given to model
    red_mean = red_resize.mean()
    red_std = red_resize.std()
    red_resize = (red_resize-red_mean) / red_std

    # convert red and green channel to torch tensors to apply the motion field later on
    red = torch.from_numpy(red[:-1, np.newaxis, :, :]).float().cuda()
    green = torch.from_numpy(green[:-1, np.newaxis, :, :]).float().cuda() if green is not None else None
    red_out = []
    green_out = [] if green is not None else None
    motion_fields = []

    # get the template frame
    if template_frame is None:
        template_frame = deepcopy(red_resize[0, np.newaxis, :, :])
    elif isinstance(template_frame, str):
        template_frame = np.load(template_frame)[0, np.newaxis, :, :]
    assert template_frame.shape == (1, model_size_y, model_size_x)
    template_frame = torch.from_numpy(template_frame).float().cuda()

    dataset = Dataset2PLight(path="", shuffle=False, batch_size=batch_size, data=red_resize)
    warper = Warper()
    upsampler = UpsamplingBilinear2d(scale_factor=(N_y/model_size_y, N_x/model_size_x))
    seed_everything(123)    
    
    # iterate through batches and apply the model to get motion field and then apply motion field to red and green channel
    for idx, batch in enumerate(dataset.train_dataloader()):
        I_t, _, I_k = batch

        if idx == 0:
            # make a template from the template frame that has the length of a batch
            template = I_k.clone().cuda()
            for i in range(batch_size):
                template[i] = template_frame
        elif I_t.shape[0] != batch_size:
            # correct length of template for last batch
            template = template[:I_t.shape[0]]
    
        # apply model to get motion field
        if upsample_derivative:
            # upsample motion field before integrating
            I_t_deform = model.get_upsampled_grid(I_t.cuda(), template.cuda())
        else:
            I_t_deform, _ = model(I_t.cuda(), template.cuda())

            # upsample motion field if necessary
            if N_y != model_size_y or N_x != model_size_x:
                I_t_deform = upsampler(I_t_deform).cuda()

        # apply (upsampled) motion field to NOT-z-scored data
        red_applied = warper(red[idx*batch_size:min((idx+1)*batch_size, N_frames - 1)], I_t_deform)
        red_applied = np.squeeze(torch_to_numpy(red_applied))
        red_out.append(red_applied)

        if green is not None:
            green_applied = warper(green[idx*batch_size:min((idx+1)*batch_size, N_frames - 1)], I_t_deform)
            green_applied = np.squeeze(torch_to_numpy(green_applied))
            green_out.append(green_applied)

        I_t_deform = np.moveaxis(torch_to_numpy(I_t_deform), 1, -1)  # make the channels to be the last dimension
        motion_fields.append(I_t_deform)

    # concatenate batches, save them and return
    red_out = np.concatenate(red_out, axis=0)
    green_out = np.concatenate(green_out, axis=0) if green is not None else None
    motion_fields = np.concatenate(motion_fields, axis=0)

    if red_out_dir is not None:
        utils2p.save_img(red_out_dir, red_out)
    if green_out_dir is not None:
        utils2p.save_img(green_out_dir, green_out)
    if motion_field_out_dir is not None:
        np.save(motion_field_out_dir, motion_fields)

    return red_out, green_out, motion_fields



if __name__ == "__main__":
    OFCO = False

    date_dir = os.path.join(load.LOCAL_DATA_DIR_LONGTERM, "210212")
    fly_dirs = load.get_flies_from_datedir(date_dir)
    trial_dirs = load.get_trials_from_fly(fly_dirs)
    trial_dir = trial_dirs[0][0]

    red = os.path.join(trial_dir, load.PROCESSED_FOLDER, load.RAW_RED_TIFF)
    green = os.path.join(trial_dir, load.PROCESSED_FOLDER, load.RAW_GREEN_TIFF)
    offset_dir = os.path.join(trial_dir, load.PROCESSED_FOLDER, "com_offset.npy")
    crop_size = (32, 48)

    red_out_dir = os.path.join(trial_dir, load.PROCESSED_FOLDER,"dn640_warped_red.tif")  # dn128_warped_red, dn640_warped_red
    green_out_dir = os.path.join(trial_dir, load.PROCESSED_FOLDER,"dn640_warped_green.tif")  # dn128_warped_green, dn640_warped_green
    """
    model = load_model(lightninglog_basedir="/home/jbraun/data/longterm/210212/Fly1/processed/lightning_logs/version_0", 
                     image_size_x=128, image_size_y=128, 
                     orig_image_size_x=640, orig_image_size_y=416, 
                     loss_factor_smooth=10)
    """
    model = load_model(lightninglog_basedir="/home/jbraun/data/longterm/210212/Fly1/processed/lightning_logs/version_9", 
                     image_size_x=640, image_size_y=416, 
                     orig_image_size_x=640, orig_image_size_y=416, 
                     loss_factor_smooth=10)
    """                
    model = load_model(lightninglog_basedir="/home/jbraun/data/longterm/210212/Fly1/processed/lightning_logs/version_13", 
                     image_size_x=128, image_size_y=128, 
                     orig_image_size_x=640, orig_image_size_y=416, 
                     loss_factor_smooth=10)
    """
    red_applied, green_applied, motion_fields = apply_model_to_trial(model=model, red=red, offset_dir=offset_dir, 
                                                      original_shape=True, upsample_derivative=False,
                                                      green=green, template_frame=None, crop_size=crop_size,
                                                      red_out_dir=red_out_dir, green_out_dir=green_out_dir,
                                                      batch_size=8)  # 8 32

    make_video_2p(green=green_applied, out_dir="/home/jbraun/tmp/deepwarp", 
                  video_name="corrected_9.mp4", red=red_applied, frames=np.arange(500),
                  trial_dir=trial_dir)

    make_video_motion_field(motion_fields=motion_fields, out_dir="/home/jbraun/tmp/deepwarp", 
                            video_name="motion_field_9.mp4", frames=np.arange(500), 
                            trial_dir=trial_dir,
                            visualisation="grid", line_distance=5, warping="dnn")
    
    if OFCO:
        motion_fields_ofco = np.load(os.path.join(trial_dir, load.PROCESSED_FOLDER, "motion_field_com.npy"))
        motion_fields_ofco = deepreg_prepare.crop_stack(motion_fields_ofco, crop_size)
        make_video_motion_field(motion_fields=motion_fields_ofco, out_dir="/home/jbraun/tmp/deepwarp", 
                                video_name="motion_field_0_ofco.mp4", frames=np.arange(500), 
                                trial_dir=trial_dir,
                                visualisation="grid", line_distance=5, warping="ofco")

    a = 0
