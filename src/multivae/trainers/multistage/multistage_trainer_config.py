from ..base import BaseTrainerConfig


class MultistageTrainerConfig(BaseTrainerConfig):
    """Configuration for a specific trainer that handles the training of the joint VAE models.

    Args:
        output_dir (str): The directory where model checkpoints, configs and final
            model will be stored. Default: None.
        per_device_train_batch_size (int): The number of training samples per batch and per device.
            Default 64
        per_device_eval_batch_size (int): The number of evaluation samples per batch and per device.
            Default 64
        num_epochs (int): The maximal number of epochs for training. Default: 100
        train_dataloader_num_workers (int): Number of subprocesses to use for train data loading.
            0 means that the data will be loaded in the main process. Default: 0
        eval_dataloader_num_workers (int): Number of subprocesses to use for evaluation data
            loading. 0 means that the data will be loaded in the main process. Default: 0
        optimizer_cls (str): The name of the `torch.optim.Optimizer` used for
            training. Default: :class:`~torch.optim.Adam`.
        optimizer_params (dict): A dict containing the parameters to use for the
            `torch.optim.Optimizer`. If None, uses the default parameters. Default: None.
        scheduler_cls (str): The name of the `torch.optim.lr_scheduler` used for
            training. If None, no scheduler is used. Default None.
        scheduler_params (dict): A dict containing the parameters to use for the
            `torch.optim.le_scheduler`. If None, uses the default parameters. Default: None.
        learning_rate (int): The learning rate applied to the `Optimizer`. Default: 1e-4
        steps_saving (int): A model checkpoint will be saved every `steps_saving` epoch.
            Default: None
        steps_predict (int): A prediction using the best model will be run every `steps_predict`
            epoch. Default: None
        keep_best_on_train (bool): Whether to keep the best model on the train set. Default: False
        seed (int): The random seed for reproducibility
        no_cuda (bool): Disable `cuda` training. Default: False
        world_size (int): The total number of process to run. Default: -1
        local_rank (int): The rank of the node for distributed training. Default: -1
        rank (int): The rank of the process for distributed training. Default: -1
        dist_backend (str): The distributed backend to use. Default: 'nccl'
        master_addr (str): The master address for distributed training. Default: 'localhost'
        master_port (str): The master port for distributed training. Default: '12345'
    """

    name_trainer = "TwoStepsTrainerConfig"
