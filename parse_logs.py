import re

with open("tensorboard/ExtraAdam/gan_ep100_lrD0.0001_lrG0.0005_bsz64_imsz64optbeta0.92022-06-11T21:04:53.149534/run.txt", "r") as fp:
    lines = fp.readlines()

    # ... read line from sw
    for line in lines:

        # define possible regex
        num=r'(\d+.?\d*)'
        log_losses = ['Loss_D:', 'Loss_G:']
        log_fid = ['IS_mean:', 'IS_std:', 'FID:']
        epochs_regex =  fr'^\[(\d*)/(\d*)\]\[(\d*)/(\d*)\]\s*'

        # define line regex based on line type
        line_regex = ""
        if 'Loss' in line:
            line_regex = epochs_regex + r'\s*'.join([fr'{t}\s*{num}' for t in log_losses])
        elif 'FID' in line:
            line_regex = epochs_regex + r'\s*'.join([fr'{t}\s*{num}' for t in log_fid])

        # Get all groups
        print(
            re.findall(line_regex, line)
        )
