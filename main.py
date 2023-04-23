import torch
from torch import autocast
import clip
from einops import rearrange
from huggingface_hub import hf_hub_download
import numpy as np
from PIL import Image

import argparse
from contextlib import nullcontext

from ldm.extras import load_model_from_config
from ldm.models.diffusion.ddim import DDIMSampler


def to_im_list(x_samples_ddim):
    x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)
    ims = []
    for x_sample in x_samples_ddim:
        x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
        ims.append(Image.fromarray(x_sample.astype(np.uint8)))
    return ims


def sample(sampler, model, c, uc, scale, start_code, h=512, w=512, precision='autocast', ddim_steps=50):
    ddim_eta = 0.0
    precision_scope = autocast if precision == 'autocast' else nullcontext
    with precision_scope('cpu'):
        shape = [4, h // 8, w // 8]
        samples_ddim, _ = sampler.sample(S=ddim_steps,
                                         conditioning=c,
                                         batch_size=c.shape[0],
                                         shape=shape,
                                         verbose=False,
                                         unconditional_guidance_scale=scale,
                                         unconditional_conditioning=uc,
                                         eta=ddim_eta,
                                         x_T=start_code)

        x_samples_ddim = model.decode_first_stage(samples_ddim)
    return to_im_list(x_samples_ddim)


def get_im_c(im, clip_model):
    prompts = preprocess(im).to(DEVICE).unsqueeze(0)
    return clip_model.encode_image(prompts).float()


def run(im1, im2, model, clip_model):
    h = w = 640

    sampler = DDIMSampler(model)

    start_code = torch.randn(2, 4, h // 8, w // 8, device=DEVICE)

    with torch.no_grad():
        cond1 = get_im_c(im1, clip_model)
        cond2 = get_im_c(im2, clip_model)
    conds = [cond1, cond2]
    conds = torch.cat(conds, dim=0).unsqueeze(0)
    conds = conds.tile(2, 1, 1)

    ims = sample(sampler, model, conds, 0 * conds, 1, start_code, ddim_steps=1)
    return ims


if __name__ == '__main__':
    DEVICE = 'cpu'
    parser = argparse.ArgumentParser()

    parser.add_argument('--content_image', type=str)
    parser.add_argument('--style_image', type=str)

    args = parser.parse_args()

    ckpt = hf_hub_download(repo_id='lambdalabs/image-mixer', filename='image-mixer-pruned.ckpt')
    config = hf_hub_download(repo_id='lambdalabs/image-mixer', filename='image-mixer-config.yaml')

    model = load_model_from_config(config, ckpt, DEVICE)
    clip_model, preprocess = clip.load('ViT-L/14', device=DEVICE)

    im1, im2 = Image.open(args.content_image), Image.open(args.style_image)

    im = run(im1, im2, model, clip_model)
    im = Image.fromarray((im.detach().numpy() * 255).astype(np.uint8))
    im.save('result.png')
