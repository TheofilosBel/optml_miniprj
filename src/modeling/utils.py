from logging import Logger
import shutil
import os.path as op
import os
import torch

logger = Logger('utils')


class CheckpointPolicy:
    SAVE_NONE=-1
    SAVE_ALL=0            # Save all models
    SAVE_BEST=1           # Save best args.nb_models2save
    SAVE_BEST_PER_EPOCH=2 # Save best args.nb_models2save per epoch


def _save_model(
    model,
    args,
    save_path
):
    model.save(save_path)
    torch.save(args, op.join(save_path, 'training_args.bin'))

def remove_checkpoint(
    base,
    name: str
) -> None:
    """ Delete a folder from the output_dir (usually a checkpoint)
    """
    try:
        shutil.rmtree(op.join(base, name))
    except OSError as e:
        print("Error removing prev checkpoints: %s - %s." % (e.filename, e.strerror))


def save_checkpoint(
    model,
    args,
    epoch,
    global_step,
    best_models = None,
    model_metric: float = None
) -> None:
    """ Save the model, the args given the epoch and the global step.
        If `args.nb_models2save` is not 0 then we keep the best models base
        on the `model_metric` usually the r@1 .
    """
    if args.checkpoint_policy == CheckpointPolicy.SAVE_NONE:
        return

    # Create a checkpoint dir for all the checkpoints
    checkpoint_base = op.join(args.checkpoint_dir, f'{args.model_checkpoint_prefix}-checkpoints')
    if not op.isdir(checkpoint_base):
        os.makedirs(checkpoint_base)

    # Create another dir for the specific cehckpoint
    specific_dir = 'cp-{}-{}'.format(epoch, global_step)
    if args.checkpoint_policy == CheckpointPolicy.SAVE_ALL:
        pass # Do nothing
    elif args.checkpoint_policy == CheckpointPolicy.SAVE_BEST:
        assert best_models != None and model_metric != None, "Need to pass best models (as empyt list) and the metric to use"
        specific_dir += f"_r@1_{model_metric:.4f}"

        best_models.append((specific_dir, model_metric))
        best_models.sort(key=lambda x: x[1], reverse=True)
        if len(best_models) > args.nb_models2save:
            model_dir_to_remove = best_models[args.nb_models2save][0]
            del best_models[args.nb_models2save]
            if model_dir_to_remove == specific_dir:
                return # Don't save this, it's the worst
            else:
                remove_checkpoint(checkpoint_base, model_dir_to_remove)
    elif args.checkpoint_policy == CheckpointPolicy.SAVE_BEST_PER_EPOCH:
        specific_dir += f"_r@1_{model_metric:.4f}"

        best_models.append((specific_dir, epoch, model_metric))
        best_m_epoch = [(d, metric, i) for i, (d, e, metric) in enumerate(best_models) if e == epoch]
        best_m_epoch.sort(key=lambda x: x[1], reverse=True)
        if len(best_m_epoch) > args.nb_models2save:
            model_dir_to_remove = best_m_epoch[args.nb_models2save]
            del best_models[model_dir_to_remove[2]]
            if model_dir_to_remove[0] == specific_dir:
                return # Don't save this, it's the worst
            else:
                remove_checkpoint(checkpoint_base, model_dir_to_remove[0])
    else:
        return

    # Create checkpoint
    os.makedirs(op.join(checkpoint_base, specific_dir))
    _save_model(model, args, op.join(checkpoint_base,specific_dir))


def load_checkpoint(
    args,
    device,
    model_class
):
    """ Load from a checkpoint the model
    """
    model_path = args.model_path
    args = torch.load(op.join(args.model_path, 'training_args.bin'))

    model = model_class(args)
    model.load(model_path)
    model.to(device)

    return model