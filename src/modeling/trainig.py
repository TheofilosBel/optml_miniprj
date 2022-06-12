from csv import writer
from typing import Any, List
import torch

import torch.optim as optim
import torchvision.utils as vutils
import functools
from torch.utils.data import DataLoader
from torch.nn.modules.loss import _Loss
from torch.autograd import Variable
from src.modeling.utils import save_checkpoint
from tqdm import tqdm

from src.modeling.utils import save_checkpoint
from src.modeling.gan import GAN
from src.modeling.train_logger import TBWritter, tb_write_fid, tb_write_metrics
from src.utils import plot_real_vs_fake_imgs, save_gen_images
from src.inception_utils import sample_gema, prepare_inception_metrics


# TODO Make loss call 2 functoins, 1 for D and 1 for G
# TODO Add WGAN-GP loss


def train(
        args,
        gan: GAN,
        dataloader: DataLoader,
        criterion: _Loss,
        optimizerD: optim.Optimizer,
        optimizerG: optim.Optimizer,
        device):
    '''
        Trains a GAN network on a specific dataset loaded by dataloader
    '''
    # Lazily initialized tensorboard writter
    tbwriter = TBWritter(args)

    print_training_args(args)

    get_inception_metrics = prepare_inception_metrics(args.inception, False)
    sample_fn = functools.partial(sample_gema, g_ema=gan.netG, device=device, nz=args.nz, batch_size=args.batch_size)

    # Create batch of latent vectors that we will use to visualize the progression of the generator
    fixed_noise = torch.randn(args.nb_fake_img, args.nz, 1, 1, device=device)
    print("Fake image noise shape:", fixed_noise.shape)

    # Establish convention for real and fake labels during training
    real_label = 1.
    fake_label = 0.

    # Lists to keep track of progress
    img_list = []
    iters = 0

    print("Starting Training Loop...")
    # For each epoch
    for epoch in tqdm(range(args.num_epochs), desc='Epochs'):
        # For each batch in the dataloader
        for i, data in tqdm(enumerate(dataloader, 0), total=len(dataloader), desc='Batch', leave=False):

            # Train both G & D
            errD, D_x, D_G_z1, label, fake = _train_D(gan, data, real_label, fake_label, optimizerD, criterion, device, args, i)
            errG, D_G_z2 = _train_G(gan, fake, real_label, label, optimizerG, criterion, args, i)

            iters += 1

            # Output training stats
            if (iters % args.nb_log_steps == 0) or ((epoch == args.num_epochs-1) and (i == len(dataloader)-1)):
                tb_write_metrics(args, tbwriter, errD.item(), errG.item(), epoch, i, iters, len(dataloader), D_x, D_G_z1, D_G_z2)

            # Check how the generator is doing by saving G's output on fixed_noise
            if (iters % args.nb_fid_log_steps == 0) or ((epoch == args.num_epochs-1) and (i == len(dataloader)-1)):
                IS_mean, IS_std, FID = get_inception_metrics(sample_fn, num_inception_images=10000, use_torch=False)
                tb_write_fid(args, tbwriter, IS_mean, IS_std, FID, epoch, i, iters, len(dataloader))

        if epoch % args.fake_img_log_interval == 0 or epoch == args.num_epochs-1:
            with torch.no_grad():
                fake = gan.netG(fixed_noise).detach().cpu()
            img_list.append(fake)

        # Save per epoch
        save_checkpoint(gan, args, epoch, iters)

    # Save in the tensorboard dir the images
    save_gen_images(args, img_list)

    # Uncomment to plot images (TODO fix makegrid)
    # real_batch = next(iter(dataloader))
    # plot_real_vs_fake_imgs(args, real_batch, img_list, device)


def _train_D(
    gan: GAN,
    data: torch.tensor,
    real_label: torch.tensor,
    fake_label: torch.tensor,
    optimizerD: torch.tensor,
    criterion,
    device,
    args, i
):
    '''
        Update D network: maximize log(D(x)) + log(1 - D(G(z)))

        in case network is wgan_gp then dont use criterion
    '''
    gan.netD.zero_grad()

    ## Train with all-real batch
    # Format batch
    real = data[0].to(device)
    b_size = real.size(0)
    label = torch.full((b_size,), real_label, dtype=torch.float, device=device)

    # Forward pass real batch through D
    output = gan.netD(real).view(-1)

    # Calculate loss on all-real batch
    if args.wgan_gp:
        errD_real = -output.mean()
    else:
        errD_real = criterion(output, label)
        # Calculate gradients for D in backward pass
        errD_real.backward()
    D_x = output.mean().item()

    ## Train with all-fake batch
    # Generate batch of latent vectors
    noise = torch.randn(b_size, args.nz, 1, 1, device=device)

    # Generate fake image batch with G
    fake = gan.netG(noise)

    # Classify all fake batch with D
    output = gan.netD(fake.detach()).view(-1)
    if args.wgan_gp:
        errD_fake = output.mean()
    else:
        label.fill_(fake_label)
        # Calculate D's loss on the all-fake batch
        errD_fake = criterion(output, label)
        # Calculate the gradients for this batch, accumulated (summed) with previous gradients
        errD_fake.backward()
    D_G_z1 = output.mean().item()

    # Compute error of D as sum over the fake and the real batches
    errD = errD_real + errD_fake

    # Backward for wgan
    if args.wgan_gp:
        errD += _grad_penalty(gan, real, fake, device, args)
        errD.backward()

    # Optimizer step
    if "ExtraAdam" == args.optim:
        if i % 2 == 0:
            optimizerD.extrapolation()
        else:
            optimizerD.step()
    else:
        optimizerD.step()


    return errD, D_x, D_G_z1, label, fake

def _train_G(
    gan: GAN,
    fake: torch.tensor,
    real_label: torch.tensor,
    label: torch.tensor,
    optimizerG: torch.tensor,
    criterion,
    args, i
):
    '''
        Update G network: maximize log(D(G(z)))
    '''

    gan.netG.zero_grad()
    label.fill_(real_label)  # fake labels are real for generator cost

    # Since we just updated D, perform another forward pass of all-fake batch through D
    # the fake is passed already through the G on the train_D step
    output = gan.netD(fake).view(-1)

    if args.wgan_gp:
         errG = - output.mean()
    else:
        # Calculate G's loss based on this output
        errG = criterion(output, label)

    # Calculate gradients for G
    errG.backward()
    D_G_z2 = output.mean().item()

    # Update G
    if "ExtraAdam" == args.optim:
        if i % 2 == 0:
            optimizerG.extrapolation()
        else:
            optimizerG.step()
    else:
        optimizerG.step()

    return errG, D_G_z2


def _grad_penalty(
    gan,
    real_data,
    fake_data,
    device,
    args
):
    batch_size = real_data.size()[0]

    # Calculate interpolation
    alpha = torch.rand(batch_size, 1, 1, 1).expand_as(real_data).to(device)

    interpolated = alpha * real_data.data + (1 - alpha) * fake_data.data
    interpolated = Variable(interpolated, requires_grad=True).to(device)

    # Calculate probability of interpolated examples
    prob_interpolated = gan.netD(interpolated)

    # Calculate grad of probabilities with respect to examples
    grad = torch.autograd.grad(
            outputs=prob_interpolated, inputs=interpolated,
            grad_outputs=torch.ones(prob_interpolated.size()).to(device),
            create_graph=True, retain_graph=True)[0]


    grad = grad.view(batch_size, -1)
    grad_l2norm = torch.sqrt(torch.sum(grad ** 2, dim=1) + 1e-12) # add epsilon to avoid probs with sqrt
    gp = torch.mean((grad_l2norm - 1) ** 2)

    # Return penalty
    return args.l_gp * gp



def print_training_args(args):
    ''' Prints some training parameters '''
    print("~~ Training Argumnets ~~")
    print(f"\tEpochs:{args.num_epochs}")
    print(f"\tlrD :{args.lr_D}")
    print(f"\tlrG :{args.lr_G}")
    print(f"\t{args.optim} beta:{args.beta1}")
    print(f"\tLogging every :{args.nb_log_steps} steps")
    print(f"\tGan args: z_size={args.nz}, g_feats_size={args.ngf}, d_feats_size={args.ndf}")