# Source: https://github.com/nv-tlabs/semanticGAN_code/blob/main/semanticGAN/prepare_inception.py

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

import argparse
from src.dataset import create_dl, create_dataset
from src import inception_utils
import pickle

@torch.no_grad()
def extract_features(args, loader, inception, device):
    pbar = loader
    pools, logits = [], []

    for data in tqdm(pbar, desc='Extract feats'):
        data = data[0].to(device)
        pool_val, logits_val = inception(data)

        pools.append(pool_val.cpu().numpy())
        logits.append(F.softmax(logits_val, dim=1).cpu().numpy())

    pools = np.concatenate(pools, axis=0)
    logits = np.concatenate(logits, axis=0)

    return pools, logits


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    parser = argparse.ArgumentParser(description='Calculate Inception v3 features for datasets')

    parser.add_argument('--dataroot', type=str, required=True, help='path to datset dir')
    parser.add_argument('--image_size', type=int, default=64, help='image dimensions (size x size)')
    parser.add_argument('--batch_size', default=64, type=int, help='batch size')
    parser.add_argument('--n_sample', type=int, default=50000)
    parser.add_argument('--output', type=str, required=True, help='path to output .pkl features file')
    parser.add_argument('--image_mode', type=str, default='RGB')
    parser.add_argument('--dataset_name', type=str, help='[celeba-mask]')
    parser.add_argument('--workers', type=int, default=8)

    args = parser.parse_args()

    inception = inception_utils.load_inception_net(device)

    # Create dataset and dataloader
    dataset = create_dataset(args)
    dataloader = create_dl(args, dataset)

    pools, logits = extract_features(args, dataloader, inception, device)

    print(f'extracted {pools.shape[0]} features')

    print('Calculating inception metrics...')
    IS_mean, IS_std = inception_utils.calculate_inception_score(logits)
    print('Training data from dataloader has IS of %5.5f +/- %5.5f' % (IS_mean, IS_std))
    print('Calculating means and covariances...')

    mean = np.mean(pools, axis=0)
    cov = np.cov(pools, rowvar=False)

    with open(args.output, 'wb') as f:
        pickle.dump({'mean': mean, 'cov': cov, 'size': args.image_size, 'path': args.dataroot}, f)