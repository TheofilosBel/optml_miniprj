import argparse

def parse_args():
    parser = argparse.ArgumentParser()

    #
    #  Optimizer parameters
    #
    parser.add_argument("--optim", type=str, required=True, help='The optimizer to use (SGD, Adam, ExtraAdam, ExtraSGD)')

    # Learning rate for optimizers
    parser.add_argument("--lr_G", type=float, default=0.0002)
    parser.add_argument("--lr_D", type=float, default=0.0002)

    # Beta1 hyperparam for Adam optimizers
    parser.add_argument("--beta1", type=float, default=0.5)
    parser.add_argument("--beta2", type=float, default=0.999)

    # Weight decay
    parser.add_argument("--wdecay", type=float, default=None)
    parser.add_argument("--eps", type=float, default=None)

    # Number of GPUs available. Use 0 for CPU mode.
    parser.add_argument("--ngpu", type=int, default=1)


    #
    #  Model parameters
    #

    # When set, model is trained as a WGAN with GP
    parser.add_argument("--wgan_gp", action="store_true")
    parser.add_argument("--l_gp", type=float, default=10)

    # Batch size during training
    parser.add_argument("--sample_size", type=str, default='mini_batch', help='full_batch for gradient descent, mini_batch for mini-batch gradient descent')
    parser.add_argument("--batch_size", type=int, default=128)

    # Spatial size of training images. All images will be resized to this size.
    parser.add_argument("--image_size", type=int, default=64)

    # Number of channels in the training images. For color images this is 3
    parser.add_argument("--nc", type=int, default=3)

    # Size of z latent vector (i.e. size of generator input)
    parser.add_argument("--nz", type=int, default=100)

    # Size of feature maps in generator
    parser.add_argument("--ngf", type=int, default=64)

    # Size of feature maps in discriminator
    parser.add_argument("--ndf", type=int, default=64)


    #
    #  Trainer params
    #
    parser.add_argument("--img_dumps_dir", type=str, default="./imgs")

    # Root directory for dataset
    parser.add_argument("--dataroot", type=str, default="./data")

    # The tensorboard dump dir
    parser.add_argument("--tb_dumps_dir", type=str, default="./tensorboard")

    # Path to inception features
    parser.add_argument("--inception", type=str , default="./inception_features/celeb_features.pkl")

    # Checkpointing args
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints")
    parser.add_argument('--checkpoint_policy', type=int, default=-1,
        help="Checkpoint policy: 0 -> Save all models | 1 -> Save nb_best_saved_models \
            2 -> Save best args.nb_best_saved_models per epoch")

    # Number of workers for dataloader
    parser.add_argument("--workers", type=int, default=8)

    # When to log
    parser.add_argument("--nb_log_steps", type=int, default=10)
    parser.add_argument("--nb_fid_log_steps", type=int, default=250, help="how often to calculate fin")
    parser.add_argument("--fake_img_log_interval", type=int, default=1, help="How often in the range of epochs should we print fake images")
    parser.add_argument("--nb_fake_img", type=int, default=4, help="How many fake images to create per interval")

    # Number of training epochs
    parser.add_argument("--num_epochs", type=int, default=50)

    return parser.parse_args()