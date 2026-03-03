def config(args):
    '''
    Hyperparameters for all models and datasets.
    '''

    '''base model'''
    args.learning_rate = 0.0001
    # emb_dim and batch_size are set via CLI (parse_args.py defaults: 200, 2048)
    # Do NOT override here — previously this silently clobbered CLI values

    '''other parameters'''
    if args.dataset == 'ENTITY':
        if args.lifelong_name == 'EWC':
            args.regular_weight = 0.1
        elif args.lifelong_name == 'SI':
            args.regular_weight = 0.01
        elif args.lifelong_name == 'LKGE':
            args.regular_weight = 0.01
            args.reconstruct_weight = 0.1
    elif args.dataset == 'RELATION':
        if args.lifelong_name == 'EWC':
            args.regular_weight = 0.1
        elif args.lifelong_name == 'SI':
            args.regular_weight = 1.0
        elif args.lifelong_name == 'LKGE':
            args.regular_weight = 0.01
            args.reconstruct_weight = 0.1
    elif args.dataset == 'FACT':
        if args.lifelong_name == 'EWC':
            args.regular_weight = 0.01
        elif args.lifelong_name == 'SI':
            args.regular_weight = 0.01
        elif args.lifelong_name == 'LKGE':
            args.regular_weight = 0.01
            args.reconstruct_weight = 1.0
    elif args.dataset == 'HYBRID':
        if args.lifelong_name == 'EWC':
            args.regular_weight = 0.01
        elif args.lifelong_name == 'SI':
            args.regular_weight = 0.01
        elif args.lifelong_name == 'LKGE':
            args.regular_weight = 0.01
            args.reconstruct_weight = 0.1



