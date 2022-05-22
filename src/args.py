import argparse


def parse_args():
    argparser = argparse.ArgumentParser()

    # The tensorboard dump dir
    argparser.add_argument("--tb_dumps_dir", type=str, default="./tensorboard")

    # Checkpointing args
    argparser.add_argument("--checkpoint_dir", type=str, default="./checkpoints")
    argparser.add_argument('--checkpoint_policy', type=int, default=0,
        help="Checkpoint policy: 0 -> Save all models | 1 -> Save nb_best_saved_models \
            2 -> Save best args.nb_best_saved_models per epoch")

    # Root directory for dataset
    argparser.add_argument("--dataroot", type=str, default="./data")

    # Number of workers for dataloader
    argparser.add_argument("--workers", type=int, default= 2)

    # When to print losses
    argparser.add_argument("--nb_log_steps", type=int, default= 10)

    # Batch size during training
    argparser.add_argument("--batch_size", type=int, default= 128)

    # Spatial size of training images. All images will be resized to this size using a transformer.
    argparser.add_argument("--image_size", type=int, default= 64)

    # Number of channels in the training images. For color images this is 3
    argparser.add_argument("--nc", type=int, default= 3)

    # Size of z latent vector (i.e. size of generator input)
    argparser.add_argument("--nz", type=int, default= 100)

    # Size of feature maps in generator
    argparser.add_argument("--ngf", type=int, default= 64)

    # Size of feature maps in discriminator
    argparser.add_argument("--ndf", type=int, default= 64)

    # Number of training epochs
    argparser.add_argument("--num_epochs", type=int, default= 5)

    # Learning rate for optimizers
    argparser.add_argument("--lr", type=int, default= 0.0002)

    # Beta1 hyperparam for Adam optimizers
    argparser.add_argument("--beta1", type=int, default= 0.5)

    # Number of GPUs available. Use 0 for CPU mode.
    argparser.add_argument("--ngpu", type=int, default= 1)

    return argparser.parse_args()