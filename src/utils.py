import matplotlib.pyplot as plt
import torchvision.utils as vutils
import numpy as np
import os.path as op
import os
import torch
import datetime as dt

def plot_img(real_batch, device):
    ''' Note blocking function '''

    img_grid = np.transpose(vutils.make_grid(real_batch[0].to(device)[:64], padding=2, normalize=True).cpu(),(1,2,0))
    plt.figure(figsize=(8,8))
    plt.axis("off")
    plt.title("Training Images")
    plt.imsave('./rand.png', img_grid.numpy())

def plot_real_vs_fake_imgs(real_batch, fake_img_list, device):
    # Plot the real images
    plt.figure(figsize=(15,15))
    plt.subplot(1,2,1)
    plt.axis("off")
    plt.title("Real Images")
    plt.imsave("./real.png",np.transpose(vutils.make_grid(real_batch[0].to(device)[:64], padding=5, normalize=True).cpu(),(1,2,0)).numpy())

    # Plot the fake images from the last epoch
    plt.subplot(1,2,2)
    plt.axis("off")
    plt.title("Fake Images")
    plt.imsave("./fake.png",np.transpose(fake_img_list[-1],(1,2,0)).numpy())
    plt.show()


def create_tb_dir(args) -> None:
    """
        Create the tensor board directory for the current training.
    """
    if not op.isdir(args.tb_dumps_dir):
        os.mkdir(args.tb_dumps_dir)

    # Format the name using some of the args given
    args.model_checkpoint_prefix = \
        f"gan_ep{args.num_epochs}_lr{args.lr}_bsz{args.batch_size}_imsz{args.image_size}" \
        f"optbeta{args.beta1}" + \
        f"{dt.datetime.now().isoformat()}"

    dumps_dir = op.join(args.tb_dumps_dir, args.model_checkpoint_prefix)
    os.mkdir(dumps_dir)
    # Dump args
    torch.save(args, op.join(dumps_dir,"config.pt"))

    return  dumps_dir

