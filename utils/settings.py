from args import Args


def get_setting(args: Args):
    if args.setting is not None:
        return args.setting
    setting = "{}_{}_{}_ft{}_sl{}_pl{}_cycle{}_{}_seed{}".format(
        args.model_id,
        args.model,
        args.data,
        args.features,
        args.seq_len,
        args.pred_len,
        args.cycle,
        args.model_type,
        args.random_seed,
    )
    if args.setting_suffix is not None:
        setting = f"{setting}_{args.setting_suffix}"
    if args.setting_prefix is not None:
        setting = f"{args.setting_prefix}_{setting}"
    return setting
