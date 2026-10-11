"""Experiment training budget; independent of routing and reward policy."""


def training_limit_reached(args, samples):
    return args.max_train_samples > 0 and samples >= args.max_train_samples


def limit_training_batch(batch, args, samples):
    if args.max_train_samples == 0:
        return batch
    # Works with the runners' lists and MMLU's positional DataFrame slice.
    return batch[:max(0, args.max_train_samples - samples)]
