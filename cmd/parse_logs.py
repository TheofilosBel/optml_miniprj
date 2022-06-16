import os
from pydoc import describe
import re
import argparse
import pandas as pd
from tqdm import tqdm
import os.path as op

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--tb_dir", required=True, type=str, help='The tensorboard dir that run_gan saves the tensorboard data')
    parser.add_argument("--out_dir", required=True, type=str, help='The dir to dump the parsed logs as csv ')
    args = parser.parse_args()
    tensorboard_dir = args.tb_dir

    if not op.isdir(tensorboard_dir):
        print("[INFO] Argument 'tb_dir' doesn't exist, we will create it")
        os.mkdir(tensorboard_dir)

    # define possible regex
    num=r'(-?[0-9]+\.?[0-9]*)'
    log_losses = ['Loss_D:', 'Loss_G:']
    log_fid = ['IS_mean:', 'IS_std:', 'FID:']
    epochs_regex =  fr'^\[(\d*)/(\d*)\]\[(\d*)/(\d*)\]\s*'

    data_fid, data_loss = [], []
    dir_list = os.listdir(tensorboard_dir)
    for optim_name in tqdm(dir_list, desc='Parsing optimizer runs', total=len(dir_list)):
        optim_dir = op.join(tensorboard_dir, optim_name)
        for run_name in os.listdir(optim_dir):
            run_dir = op.join(optim_dir, run_name)

            # Get info from run_name
            matches = re.findall(rf'lrD{num}_lrG{num}_bsz{num}_.+?beta{num}', run_name)
            if len(matches) == 0:
                matches = re.findall(rf'lr{num}_bsz{num}_.+?beta{num}', run_name)
                matches = [(matches[0][0], ) + matches[0]] # we had 1 lr for d and g

            lrD, lrG, bsz, beta = matches[0]
            loss = 'Wgangp' if 'wgan' in run_name else 'St'
            line_dict = {'optimizer': optim_name,
                        "lrD": lrD, "lrG": lrG, "bsz":bsz,
                        "beta": beta, 'loss': loss,
                        "name": f"{optim_name}_lrG{lrG}_lrD{lrD}_beat{beta}_bsz{bsz}_loss{loss}" }


            # Read the run.txt with the logs
            with open(f"{run_dir}/run.txt", "r") as fp:
                for line in fp:

                    # define line regex based on line type
                    line_regex = ""
                    if 'Loss' in line:
                        line_regex = epochs_regex + r'\s*'.join([fr'{t}\s*{num}' for t in log_losses])
                        matches = re.findall(line_regex, line)
                        cur_e, max_e, cur_itr, max_iter, lossD, lossG =\
                            matches[0] if len(matches) != 0 else (None,) * 6
                        data_loss.append({ **line_dict,
                            'cur_epoch': int(cur_e), 'max_epoch': int(max_e),
                            'cur_iter': int(cur_itr), 'max_iter': int(max_iter),
                            'lossD': float(lossD), 'lossG': float(lossG)
                        })
                    elif 'FID' in line:
                        line_regex = epochs_regex + r'\s*'.join([fr'{t}\s*{num}' for t in log_fid])
                        matches = re.findall(line_regex, line)
                        cur_e, max_e, cur_itr, max_iter, IS_mean, IS_std, FID =\
                            matches[0] if len(matches) != 0 else (None,) * 7
                        data_fid.append({**line_dict,
                            'cur_epoch': int(cur_e), 'max_epoch': int(max_e),
                            'cur_iter': int(cur_itr), 'max_iter': int(max_iter),
                            'IS_mean': float(IS_mean), 'IS_std': float(IS_std), 'FID':float(FID)
                        })

    print("Saving to csv..")
    loss_df = pd.DataFrame(data_loss)
    print("Print a sample from parsing results..")
    print(loss_df.head())
    loss_df['iter'] = loss_df['cur_epoch']*loss_df['max_iter'] + loss_df['cur_iter']

    fid_df = pd.DataFrame(data_fid)
    print(fid_df.head())
    fid_df['iter'] = fid_df['cur_epoch']*fid_df['max_iter'] + fid_df['cur_iter']

    # Save to output
    if not op.isdir(args.out_dir):
        os.mkdir(args.out_dir)

    loss_df.to_csv(op.join(args.out_dir, 'loss_df.csv'))
    fid_df.to_csv(op.join(args.out_dir, 'fid_df.csv'))

    print("Done")
