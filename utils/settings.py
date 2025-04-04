from args import Args


def get_setting(args: Args):
    setting = '{}_{}_{}_ft{}_sl{}_pl{}_cycle{}_{}_seed{}'.format(
        args.model_id,
        args.model,
        args.data,
        args.features,
        args.seq_len,
        args.pred_len,
        args.cycle,
        args.model_type,
        args.random_seed)
    return setting
