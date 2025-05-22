import os
import shutil
import tempfile
from copy import deepcopy

import pytest
import torch
from pytest import fixture, mark
from torch import nn

from multivae.data import IncompleteDataset, MultimodalBaseDataset
from multivae.models.auto_model import AutoModel
from multivae.models.base import ModelOutput
from multivae.models.mhvae import MHVAE, MHVAEConfig
from multivae.trainers import BaseTrainer, BaseTrainerConfig

from .mhvae_test_architectures import (
    add_bu,
    bu_1,
    bu_2,
    my_input_decoder,
    my_input_encoder,
    posterior_block,
    prior_block,
    td_1,
    td_2,
)


class Test_MHVAE:
    """Test the MHVAE class."""

    @fixture(params=["complete", "incomplete"])
    def dataset(self, request):
        """Create a dummy dataset for testing"""
        if request.param == "complete":
            return MultimodalBaseDataset(
                data=dict(
                    m1=torch.randn((100, 3, 28, 28)), m0=torch.randn((100, 3, 28, 28))
                )
            )
        else:
            return IncompleteDataset(
                data=dict(
                    m1=torch.randn((100, 3, 28, 28)), m0=torch.randn((100, 3, 28, 28))
                ),
                masks={
                    "m1": torch.Tensor([True] * 50 + [False] * 50),
                    "m0": torch.Tensor([True] * 100),
                },
            )

    @fixture(
        params=[[("normal", "normal"), 3, 1.0, 10], [("normal", "laplace"), 4, 2.5, 15]]
    )
    def model_config(self, request):
        """Create the model configuration"""
        return MHVAEConfig(
            n_modalities=2,
            latent_dim=request.param[3],
            decoders_dist=dict(m0=request.param[0][0], m1=request.param[0][1]),
            n_latent=request.param[1],
            beta=request.param[2],
        )

    @fixture(params=[True, False])
    def wn(self, request):
        """Test with and without weight_norm in the architectures."""
        return request.param

    @fixture(params=[True, False])
    def architectures(self, model_config, wn, request):
        """Instantiate architectures for testing the model class."""
        encoders = dict(m0=my_input_encoder(), m1=my_input_encoder())
        decoders = dict(m0=my_input_decoder(), m1=my_input_decoder())
        bottom_up_blocks = dict(m0=[bu_1()], m1=[bu_1()])

        if model_config.n_latent == 4:
            bottom_up_blocks["m0"].append(add_bu())
            bottom_up_blocks["m1"].append(add_bu())

        bottom_up_blocks["m0"].append(bu_2(64, 128, model_config.latent_dim))
        bottom_up_blocks["m1"].append(bu_2(64, 128, model_config.latent_dim))

        td_blocks = [td_1()]
        if model_config.n_latent == 4:
            td_blocks.append(add_bu())

        td_blocks.append(td_2(model_config.latent_dim))

        if model_config.n_latent == 4:
            prior_blocks = [
                prior_block(32, wn),
                prior_block(64, wn),
                prior_block(64, wn),
            ]
            if request.param:
                posterior_blocks = [
                    posterior_block(32, wn),
                    posterior_block(64, wn),
                    posterior_block(64, wn),
                ]
            else:
                posterior_blocks = {}
                for m in encoders:
                    posterior_blocks[m] = [
                        posterior_block(32, wn),
                        posterior_block(64, wn),
                        posterior_block(64, wn),
                    ]

        else:
            prior_blocks = [prior_block(32, wn), prior_block(64, wn)]
            if request.param:
                posterior_blocks = [posterior_block(32, wn), posterior_block(64, wn)]
            else:
                posterior_blocks = {}
                for m in encoders:
                    posterior_blocks[m] = [
                        posterior_block(32, wn),
                        posterior_block(64, wn),
                    ]

        return dict(
            encoders=encoders,
            decoders=decoders,
            bottom_up_blocks=bottom_up_blocks,
            top_down_blocks=td_blocks,
            prior_blocks=prior_blocks,
            posterior_blocks=posterior_blocks,
        )

    def test_setup(self, model_config, architectures):
        """Test the model init. Check attributes and modules."""
        model = MHVAE(model_config=model_config, **architectures)

        assert model.latent_dim == model_config.latent_dim
        assert model.beta == model_config.beta
        assert model.n_latent == model_config.n_latent

        assert isinstance(model.encoders, nn.ModuleDict)
        assert isinstance(model.decoders, nn.ModuleDict)
        assert isinstance(model.bottom_up_blocks, nn.ModuleDict)
        assert isinstance(model.top_down_blocks, nn.ModuleList)
        assert isinstance(model.prior_blocks, nn.ModuleList)
        if model.share_posterior_weights:
            assert isinstance(model.posterior_blocks, nn.ModuleList)
        else:
            assert isinstance(model.posterior_blocks, nn.ModuleDict)

        return

    @pytest.fixture
    def model(self, model_config, architectures):
        """Create model for testing"""
        return MHVAE(model_config=model_config, **architectures)

    def test_sanity_check_bottom_up(self, model):
        """Test the sanity_check_bottom_up method.
        We check that the method raises an error when the bottom up blocks don't have
        matching keys with the encoders.
        We check that an error is raised when the lenght of the bottom up blocks don't match between modalities.
        """
        wrong_bottom_up = deepcopy(model.bottom_up_blocks)
        with pytest.raises(AttributeError):
            model.sanity_check_bottom_up(model.encoders, wrong_bottom_up.pop("m0"))

        wrong_bottom_up = deepcopy(model.bottom_up_blocks)
        wrong_bottom_up["m2"] = wrong_bottom_up.pop("m1")

        with pytest.raises(AttributeError):
            model.sanity_check_bottom_up(model.encoders, wrong_bottom_up)

        wrong_bottom_up = deepcopy(model.bottom_up_blocks)
        wrong_bottom_up["m0"] = wrong_bottom_up["m0"][:-1]
        with pytest.raises(AttributeError):
            model.sanity_check_bottom_up(model.encoders, wrong_bottom_up)

        # Check that an error is raised when the last block is not an instance
        # of BaseEncoder.
        wrong_bottom_up = deepcopy(model.bottom_up_blocks)
        wrong_bottom_up["m0"][-1] = wrong_bottom_up["m0"][-2]
        with pytest.raises(AttributeError):
            model.sanity_check_bottom_up(model.encoders, wrong_bottom_up)

        return

    def test_sanity_check_top_down(self, model):
        """Test the sanity_check_top_down method.
        We check that the method raises an error when the top down blocks don't have
        matching lenghts between modalities.
        """
        wrong_top_bottom = deepcopy(model.top_down_blocks)
        wrong_top_bottom = wrong_top_bottom[:-1]
        with pytest.raises(AttributeError):
            model.sanity_check_top_down_blocks(wrong_top_bottom)
        return

    def test_check_and_set_posterior_blocks(self, model):
        """Test the check_and_set_posterior_blocks method.
        We check that the method raises an error when the posterior blocks don't have
        matching keys with the encoders or the right number of blocks.
        """
        wrong_posteriors = deepcopy(model.posterior_blocks)
        if isinstance(wrong_posteriors, nn.ModuleList):
            wrong_posteriors = wrong_posteriors[:-1]
        else:
            wrong_posteriors["m2"] = wrong_posteriors.pop("m1")
        with pytest.raises(AttributeError):
            model.check_and_set_posterior_blocks(wrong_posteriors)

        wrong_posteriors = deepcopy(model.posterior_blocks)
        if isinstance(wrong_posteriors, nn.ModuleList):
            wrong_posteriors[-1] = nn.Linear(2, 3)
        else:
            wrong_posteriors["m1"] = wrong_posteriors.pop("m1")[:-1]
        with pytest.raises(AttributeError):
            model.check_and_set_posterior_blocks(wrong_posteriors)

        wrong_posteriors = deepcopy(model.posterior_blocks)
        if isinstance(wrong_posteriors, nn.ModuleDict):
            wrong_posteriors["m1"][-1] = nn.Linear(2, 3)
            with pytest.raises(AttributeError):
                model.check_and_set_posterior_blocks(wrong_posteriors)

    def test_sanity_check_prior_blocks(self, model):
        """Test the sanity_check_prior_blocks method.
        We check that the method raises an error when the prior blocks don't have
        the right number of blocks or the right type of blocks.
        Each prior block should be an instance of BaseEncoder.
        """
        wrong_priors = deepcopy(model.prior_blocks)
        wrong_priors = wrong_priors[:-1]
        with pytest.raises(AttributeError):
            model.sanity_check_prior_blocks(wrong_priors)

        wrong_priors = deepcopy(model.prior_blocks)
        wrong_priors[-1] = torch.nn.Linear(2, 3)
        with pytest.raises(AttributeError):
            model.sanity_check_prior_blocks(wrong_priors)

        return

    def test_model_without_architectures(self, model_config, architectures):
        """The MHVAE model cannot be initialized without encoders or decoders.
        We check that the model raises an error when the encoders or decoders are not provided.
        """
        with pytest.raises(TypeError):
            archi = deepcopy(architectures)
            archi.pop("encoders")
            MHVAE(model_config=model_config, **archi)
        with pytest.raises(TypeError):
            architectures.pop("decoders")
            MHVAE(model_config=model_config, **architectures)

    def test_forward(self, model, dataset):
        """Test the forward method of the model.
        We check that the method returns a ModelOutput with the right shape and attributes.
        """
        samples = dataset[:10]
        output = model(samples)

        assert isinstance(output, ModelOutput)
        assert hasattr(output, "loss")
        assert isinstance(output.loss, torch.Tensor)

        return

    def test_encode(self, dataset, model):
        """Test the encode method of the model.
        We check that the method returns a ModelOutput with the right shape and attributes.
        """
        for return_mean in [True, False]:
            samples = dataset[:10]

            output = model.encode(samples, return_mean=return_mean)
            assert isinstance(output, ModelOutput)
            assert hasattr(output, "z")
            assert output.z.shape == (10, 32, 14, 14)
            assert hasattr(output, "all_z")
            assert output.one_latent_space
            assert isinstance(output.all_z, dict)

            output = model.encode(samples, N=4, return_mean=return_mean)
            assert isinstance(output, ModelOutput)
            assert output.z.shape == (4, 10, 32, 14, 14)
            assert hasattr(output, "all_z")
            assert output.one_latent_space
            assert isinstance(output.all_z, dict)

            output = model.encode(samples, N=4, flatten=True, return_mean=return_mean)
            assert isinstance(output, ModelOutput)
            assert output.z.shape == (4 * 10, 32, 14, 14)
            assert hasattr(output, "all_z")

    def test_decode(self, model, dataset):
        """Test the decode method of the model.
        We check that the method returns a ModelOutput with the right shape and attributes.
        """
        samples = dataset[:10]

        embeddings = model.encode(samples)
        output = model.decode(embeddings)

        assert isinstance(output, ModelOutput)
        assert hasattr(output, "m0")
        assert output.m0.shape == samples.data["m0"].shape

        return

    def test_predict(self, model, dataset):
        """Test the predict method of the model.
        We check that the method returns a ModelOutput with the right shape and attributes.
        """
        samples = dataset[:10]

        # Test reconstruction
        output = model.predict(cond_mod="m0", gen_mod="m1", inputs=samples)
        assert isinstance(output, ModelOutput)
        assert hasattr(output, "m1")
        assert output.m1.shape == samples.data["m1"].shape

        output = model.predict(cond_mod="m1", gen_mod="m0", inputs=samples, N=10)
        assert isinstance(output, ModelOutput)
        assert hasattr(output, "m0")
        assert output.m0.shape == (10, 10, 3, 28, 28)

        output = model.predict(cond_mod=["m0"], gen_mod="m1", inputs=samples)
        assert isinstance(output, ModelOutput)
        assert hasattr(output, "m1")
        assert output.m1.shape == samples.data["m1"].shape

        output = model.predict(cond_mod=["m0", "m1"], gen_mod="all", inputs=samples)
        assert isinstance(output, ModelOutput)
        assert hasattr(output, "m0")
        assert hasattr(output, "m1")

        assert output.m0.shape == samples.data["m0"].shape
        assert output.m1.shape == samples.data["m1"].shape

        return

    def test_no_gradient_towards_missing_mods(self, model, dataset):
        """Check that gradients are null for missing modalities."""
        if hasattr(dataset, "masks"):
            # Compute loss on incomplete data
            loss = model(dataset[50:]).loss.sum()

            loss.backward()

            assert all(
                [
                    torch.all(parameter.grad == 0)
                    for parameter in model.encoders["m1"].parameters()
                ]
            )
            assert all(
                [
                    torch.all(parameter.grad == 0)
                    for parameter in model.decoders["m1"].parameters()
                ]
            )

            # Compute loss on complete data
            loss = model(dataset[:50]).loss.sum()

            loss.backward()

            assert not all(
                [
                    torch.all(parameter.grad == 0)
                    for parameter in model.encoders["m1"].parameters()
                ]
            )
            assert not all(
                [
                    torch.all(parameter.grad == 0)
                    for parameter in model.decoders["m1"].parameters()
                ]
            )

    @fixture(params=[[32, 64, 3, "Adagrad"], [16, 16, 4, "Adam"]])
    def trainer_config(self, request):
        """Create a trainer configuration for testing."""
        tmp = tempfile.mkdtemp()

        yield BaseTrainerConfig(
            output_dir=tmp,
            per_device_eval_batch_size=request.param[0],
            per_device_train_batch_size=request.param[1],
            num_epochs=request.param[2],
            optimizer_cls=request.param[3],
            learning_rate=1e-4,
            steps_saving=2,
        )
        shutil.rmtree(tmp)

    @fixture
    def trainer(self, trainer_config, model, dataset):
        """Create a trainer for testing."""
        return BaseTrainer(
            model,
            train_dataset=dataset,
            eval_dataset=dataset,
            training_config=trainer_config,
        )

    @mark.slow
    def test_train_step(self, trainer):
        """Test train step with the MHVAE model.
        The weights should be updated after the step.
        """
        start_model_state_dict = deepcopy(trainer.model.state_dict())

        _ = trainer.train_step(epoch=1)

        step_1_model_state_dict = deepcopy(trainer.model.state_dict())

        # check that weights were updated
        assert not all(
            [
                torch.equal(start_model_state_dict[key], step_1_model_state_dict[key])
                for key in start_model_state_dict.keys()
            ]
        )

    @mark.slow
    def test_eval_step(self, trainer):
        """Test eval step with the MHVAE model.
        The weights should not be updated after the step.
        """
        start_model_state_dict = deepcopy(trainer.model.state_dict())

        _ = trainer.eval_step(epoch=1)

        step_1_model_state_dict = deepcopy(trainer.model.state_dict())

        # check that weights were not updated
        assert all(
            [
                torch.equal(start_model_state_dict[key], step_1_model_state_dict[key])
                for key in start_model_state_dict.keys()
            ]
        )

    @mark.slow
    def test_main_train_loop(self, trainer):
        """Test the main training loop with the MHVAE model.
        The weights should be updated training
        """
        start_model_state_dict = deepcopy(trainer.model.state_dict())

        trainer.train()

        step_1_model_state_dict = deepcopy(trainer.model.state_dict())

        # check that weights were updated
        assert not all(
            [
                torch.equal(start_model_state_dict[key], step_1_model_state_dict[key])
                for key in start_model_state_dict.keys()
            ]
        )

    @mark.slow
    def test_checkpoint_saving(self, model, trainer, trainer_config, wn):
        """Test checkpoint saving with the MHVAE model.
        We check that the files are saved to the right directory and that the model
        and optimizer state dicts are saved correctly and can be reloaded.
        """
        dir_path = trainer_config.output_dir

        # Make a training step, save the model and reload it
        trainer.train_step(epoch=1)

        model = deepcopy(trainer.model)
        optimizer = deepcopy(trainer.optimizer)

        trainer.save_checkpoint(dir_path=dir_path, epoch=1, model=model)

        checkpoint_dir = os.path.join(dir_path, "checkpoint_epoch_1")

        assert os.path.isdir(checkpoint_dir)

        files_list = os.listdir(checkpoint_dir)

        assert set(["model.pt", "optimizer.pt", "training_config.json"]).issubset(
            set(files_list)
        )
        if not wn:
            # check pickled custom architectures are in the checkpoint folder
            for archi in model.model_config.custom_architectures:
                assert archi + ".pkl" in files_list

        model_rec_state_dict = torch.load(os.path.join(checkpoint_dir, "model.pt"))[
            "model_state_dict"
        ]

        assert all(
            [
                torch.equal(
                    model_rec_state_dict[key].cpu(), model.state_dict()[key].cpu()
                )
                for key in model.state_dict().keys()
            ]
        )

        # Reload full model and check that it is the same
        if not wn:
            model_rec = AutoModel.load_from_folder(os.path.join(checkpoint_dir))

            assert all(
                [
                    torch.equal(
                        model_rec.state_dict()[key].cpu(), model.state_dict()[key].cpu()
                    )
                    for key in model.state_dict().keys()
                ]
            )

            assert isinstance(model_rec.encoders.cpu(), type(model.encoders.cpu()))
            assert isinstance(model_rec.decoders.cpu(), type(model.decoders.cpu()))

        optim_rec_state_dict = torch.load(os.path.join(checkpoint_dir, "optimizer.pt"))

        assert all(
            [
                dict_rec == dict_optimizer
                for (dict_rec, dict_optimizer) in zip(
                    optim_rec_state_dict["param_groups"],
                    optimizer.state_dict()["param_groups"],
                )
            ]
        )

        assert all(
            [
                dict_rec == dict_optimizer
                for (dict_rec, dict_optimizer) in zip(
                    optim_rec_state_dict["state"], optimizer.state_dict()["state"]
                )
            ]
        )

    @mark.slow
    def test_checkpoint_saving_during_training(
        self, model, trainer, trainer_config, wn
    ):
        """Test the creation of checkpoints in the main train loop.
        Check the directory structure and the files.
        """
        target_saving_epoch = trainer_config.steps_saving

        dir_path = trainer_config.output_dir

        model = deepcopy(trainer.model)

        trainer.train()

        training_dir = os.path.join(
            dir_path, f"MHVAE_training_{trainer._training_signature}"
        )
        assert os.path.isdir(training_dir)

        checkpoint_dir = os.path.join(
            training_dir, f"checkpoint_epoch_{target_saving_epoch}"
        )

        assert os.path.isdir(checkpoint_dir)

        files_list = os.listdir(checkpoint_dir)

        # check files
        assert set(["model.pt", "optimizer.pt", "training_config.json"]).issubset(
            set(files_list)
        )
        if not wn:
            # check pickled custom architectures
            for archi in model.model_config.custom_architectures:
                assert archi + ".pkl" in files_list

        model_rec_state_dict = torch.load(os.path.join(checkpoint_dir, "model.pt"))[
            "model_state_dict"
        ]

        assert not all(
            [
                torch.equal(model_rec_state_dict[key], model.state_dict()[key])
                for key in model.state_dict().keys()
            ]
        )

    @mark.slow
    def test_final_model_saving(self, model, trainer, trainer_config, wn):
        """Test final model saving after training for the MHVAE model.
        We check that the model is correctly saved and can be reloaded.
        """
        dir_path = trainer_config.output_dir

        trainer.train()

        model = deepcopy(trainer._best_model)

        training_dir = os.path.join(
            dir_path, f"MHVAE_training_{trainer._training_signature}"
        )
        assert os.path.isdir(training_dir)

        final_dir = os.path.join(training_dir, "final_model")
        assert os.path.isdir(final_dir)

        files_list = os.listdir(final_dir)

        assert set(["model.pt", "model_config.json", "training_config.json"]).issubset(
            set(files_list)
        )
        if not wn:
            # check pickled custom architectures
            for archi in model.model_config.custom_architectures:
                assert archi + ".pkl" in files_list

            # check reload full model
            model_rec = AutoModel.load_from_folder(os.path.join(final_dir))

            assert all(
                [
                    torch.equal(
                        model_rec.state_dict()[key].cpu(), model.state_dict()[key].cpu()
                    )
                    for key in model.state_dict().keys()
                ]
            )

            assert isinstance(model_rec.encoders.cpu(), type(model.encoders.cpu()))
            assert isinstance(model_rec.decoders.cpu(), type(model.decoders.cpu()))
