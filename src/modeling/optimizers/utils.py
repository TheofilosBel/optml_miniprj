def update_optim_args(args, base_args):
    ''' Update the optimizer args depending on the optimizer type'''
    if 'adam' in args.optim.lower():
        # Non optional
        base_args['betas'] = (args.beta1, args.beta2)
        # Optional
        if args.wdecay != None: base_args['weight_decay'] = args.wdecay
        if args.eps != None: base_args['eps'] = args.eps

    # No need to return, but its better to be like this
    return base_args