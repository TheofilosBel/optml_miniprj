from __future__ import print_function
#%matplotlib inline
import torch
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms


# We can use an image folder dataset the way we have it setup.
# Create the dataset
def create_dataset(args) -> dset:
    return  dset.ImageFolder(
        root=args.dataroot,
        transform=transforms.Compose([
            transforms.Resize(args.image_size),
            transforms.CenterCrop(args.image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]))

# Create the dataloader
def create_dl(args, dataset: dset) -> torch.utils.data.DataLoader:
    return torch.utils.data.DataLoader(dataset, batch_size=args.batch_size,
        shuffle=True, num_workers=args.workers)

