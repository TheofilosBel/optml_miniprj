import matplotlib.pyplot as plt
import torchvision.utils as vutils
import numpy as np
import os.path as op
import os
import torch
import datetime as dt
from typing import List

def plot_img(args, real_batch, device):
    ''' Note blocking function '''
    if not op.isdir(args.img_dumps_dir):
        os.mkdir(args.img_dumps_dir)

    img_grid = np.transpose(vutils.make_grid(real_batch[0].to(device)[:64], padding=2, normalize=True).cpu(),(1,2,0))
    plt.figure(figsize=(8,8))
    plt.axis("off")
    plt.title("Training Images")
    plt.imsave(op.join(args.img_dumps_dir, "test.png") , img_grid.numpy())

def plot_real_vs_fake_imgs(args, real_batch, fake_img_list, device):
    if not op.isdir(args.img_dumps_dir):
        os.mkdir(args.img_dumps_dir)


    # Plot the real images
    plt.figure(figsize=(15,15))
    plt.subplot(1,2,1)
    plt.axis("off")
    plt.title("Real Images")
    plt.imsave(op.join(args.img_dumps_dir, "real.png"),
        np.transpose(vutils.make_grid(real_batch[0].to(device)[:64], padding=5, normalize=True).cpu(),(1,2,0)).numpy())

    # Plot the fake images from the last epoch
    plt.subplot(1,2,2)
    plt.axis("off")
    plt.title("Fake Images")
    plt.imsave(op.join(args.img_dumps_dir, "fake.png"), np.transpose(fake_img_list[-1],(1,2,0)).numpy())
    plt.show()


def create_tb_dir(args) -> None:
    """
        Create the tensor board directory for the current training.
    """
    if not op.isdir(args.tb_dumps_dir):
        os.mkdir(args.tb_dumps_dir)

    # Create one folder per optimizer
    base_path = op.join(args.tb_dumps_dir, args.optim)
    if not op.isdir(base_path):
        os.mkdir(base_path)

    # Check if args contain the "ConfigurableParams" field that will overwrite the usual dir name
    if hasattr(args, "config_params") and isinstance(args.config_params, dict):
        config_params_str = "_".join([f"{k}{v}" for k,v in args.config_params.items()])

        # Format the name using some of the args given
        args.model_checkpoint_prefix = \
            f"exps_gan_ep{args.num_epochs}" + \
             "_" + config_params_str +  \
            f"{dt.datetime.now().isoformat()}"
    else:
        # Format the name using some of the args given
        args.model_checkpoint_prefix = \
            f"gan_ep{args.num_epochs}_lrD{args.lr_D}_lrG{args.lr_G}_bsz{args.batch_size}_imsz{args.image_size}" \
            f"optbeta{args.beta1}" + \
            f"{dt.datetime.now().isoformat()}"


    dumps_dir = op.join(base_path, args.model_checkpoint_prefix)
    os.mkdir(dumps_dir)
    torch.save(args, op.join(dumps_dir, 'args.pt'))

    return dumps_dir


def param_generator(param_list: List):
    '''
    Inputs: A list of Triplets like:
        [ ('param name', 'param_type', [val1, val2, ...]) , .... ]

        * Param_name should match the actual argument name
        * param_type can be 'opt' or 'model'
        * List[] of values can be whatever values the model or opt param permits

    Returns: A tuple with 2 maps (1 for optimizer and 1 for model):
        ({opt_param_name: val, ...}, {model_param_name: val, ...})
    '''
    indices = [0 for _ in range(len(param_list))]
    stop = False
    while not stop:

        # Extract params, return 2 dicts (opt params, model params)
        opt_params = {}
        model_params = {}
        for i, tup in enumerate(param_list):
            if tup[1] == 'opt':
                opt_params[tup[0]] = tup[2][indices[i]]
            else:
                model_params[tup[0]] = tup[2][indices[i]]
        yield opt_params, model_params

        # Advance the last index
        # if reached size advance prev index, until you reach idx = 0
        for idx in range(len(indices) - 1, -1, -1):
            if indices[idx] + 1 == len(param_list[idx][2]):
                # if idx = 0 stop, else reset all indexes from this point on
                if idx == 0: stop = True
                else:
                    for x in range(idx,len(indices)): indices[x] = 0
                continue
            else:
                indices[idx] += 1
                break
