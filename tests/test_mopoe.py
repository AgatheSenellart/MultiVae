import os
import shutil
from copy import deepcopy

import pytest
import torch
from pythae.models.base.base_utils import ModelOutput

from multivae.data.datasets.base import IncompleteDataset, MultimodalBaseDataset
from multivae.models.auto_model.auto_model import AutoModel
from multivae.models.base.base_config import BaseAEConfig
from multivae.models.mopoe import MoPoE, MoPoEConfig
from multivae.models.nn.default_architectures import Decoder_AE_MLP
from multivae.trainers.base.base_trainer import BaseTrainer
from multivae.trainers.base.base_trainer_config import BaseTrainerConfig

from .encoders import EncoderTest, EncoderTestMultilatents


class TestMoPoE:
    """Test class for MoPoE model."""

    @pytest.fixture(params=["complete", "incomplete"])
    def dataset(self, request):
        """Create basic dataset"""
        data = dict(
            mod1=torch.randn((6, 2)),
            mod2=torch.randn((6, 3)),
            mod3=torch.randn((6, 4)),
            mod4=torch.randn((6, 4)),
        )
        if request.param == "complete":
            dataset = MultimodalBaseDataset(data)
        else:
            masks = dict(
                mod1=torch.Tensor([True] * 3 + [False] * 3),
                mod2=torch.Tensor([True] * 6),
                mod3=torch.Tensor([True] * 6),
                mod4=torch.Tensor([True] * 6),
            )
            dataset = IncompleteDataset(data=data, masks=masks)

        return dataset

    @pytest.fixture(params=["one_latent_space", "multi_latent_spaces"])
    def archi_and_config(self, beta, request):
        """Create model configurations for test"""
        if request.param == "one_latent_space":
            # Create an instance of mvae model
            config1 = BaseAEConfig(input_dim=(2,), latent_dim=5)
            config2 = BaseAEConfig(input_dim=(3,), latent_dim=5)
            config3 = BaseAEConfig(input_dim=(4,), latent_dim=5)
            config4 = BaseAEConfig(input_dim=(4,), latent_dim=5)

            encoders = dict(
                mod1=EncoderTest(config1),
                mod2=EncoderTest(config2),
                mod3=EncoderTest(config3),
                mod4=EncoderTest(config4),
            )

            model_config = MoPoEConfig(
                n_modalities=4,
                latent_dim=5,
                input_dims=dict(mod1=(2,), mod2=(3,), mod3=(4,), mod4=(4,)),
                beta=beta,
            )

            decoders = dict(
                mod1=Decoder_AE_MLP(config1),
                mod2=Decoder_AE_MLP(config2),
                mod3=Decoder_AE_MLP(config3),
                mod4=Decoder_AE_MLP(config4),
            )

        else:
            config1 = BaseAEConfig(input_dim=(2,), latent_dim=5, style_dim=1)
            config2 = BaseAEConfig(input_dim=(3,), latent_dim=5, style_dim=2)
            config3 = BaseAEConfig(input_dim=(4,), latent_dim=5, style_dim=3)
            config4 = BaseAEConfig(input_dim=(4,), latent_dim=5, style_dim=3)

            encoders = dict(
                mod1=EncoderTestMultilatents(config1),
                mod2=EncoderTestMultilatents(config2),
                mod3=EncoderTestMultilatents(config3),
                mod4=EncoderTestMultilatents(config4),
            )
            model_config = MoPoEConfig(
                n_modalities=4,
                latent_dim=5,
                input_dims=dict(mod1=(2,), mod2=(3,), mod3=(4,), mod4=(4,)),
                beta=beta,
                modalities_specific_dim=dict(mod1=1, mod2=2, mod3=3, mod4=3),
            )
            decoders = dict(
                mod1=Decoder_AE_MLP(BaseAEConfig(input_dim=(2,), latent_dim=6)),
                mod2=Decoder_AE_MLP(BaseAEConfig(input_dim=(3,), latent_dim=7)),
                mod3=Decoder_AE_MLP(BaseAEConfig(input_dim=(4,), latent_dim=8)),
                mod4=Decoder_AE_MLP(BaseAEConfig(input_dim=(4,), latent_dim=8)),
            )

        return dict(encoders=encoders, decoders=decoders, model_config=model_config)

    @pytest.fixture(params=[1.0, 1.5, 2.0])
    def beta(self, request):
        """Test the model with different beta values"""
        beta = request.param

        return beta

    @pytest.fixture(params=[True, False])
    def custom(self, request):
        """Test the model with or without custom architectures"""
        return request.param

    @pytest.fixture
    def model(self, archi_and_config, custom):
        """Create the model for testing"""
        if custom:
            model = MoPoE(**archi_and_config)
        else:
            model = MoPoE(archi_and_config["model_config"])
        return model

    def test_init(self, model, custom, archi_and_config):
        """Test the MoPoE initialization."""
        # check that our custom architectures were passed to the model
        model_config = archi_and_config["model_config"]
        if custom:
            assert "encoders" in model.model_config.custom_architectures
            assert "decoders" in model.model_config.custom_architectures

            if model_config.modalities_specific_dim is not None:
                assert isinstance(model.encoders["mod1"], EncoderTestMultilatents)
            else:
                assert isinstance(model.encoders["mod1"], EncoderTest)

        # test that the subsets were set as expected
        expected_subsets = [
            ["mod1", "mod2"],
            ["mod2", "mod3"],
            ["mod1", "mod4"],
            ["mod1", "mod2", "mod3"],
            ["mod1", "mod2", "mod4"],
        ]
        for s in expected_subsets:
            assert s in list(model.subsets.values())

    def test_forward(self, model, dataset):
        """Test the MoPoE forward function.
        We check that the output is a ModelOutput with the correct tensor shapes inside.
        """
        output = model(dataset, epoch=2)
        loss = output.loss
        assert isinstance(loss, torch.Tensor)
        assert loss.size() == torch.Size([])
        assert loss.requires_grad

        # test that setting a wrong architectures raises an error in forward
        if model.model_config.modalities_specific_dim is not None:
            model.encoders["mod1"] = EncoderTest(
                BaseAEConfig(input_dim=(2,), latent_dim=5)
            )
            with pytest.raises(AttributeError):
                output = model(dataset, epoch=2)

    def test_encode(self, model, dataset, archi_and_config):
        """Test the encode method of the MoPoe."""
        latent_dim = archi_and_config["model_config"].latent_dim
        outputs = model.encode(dataset[0])
        # Check the value of 'one_latent_space'
        if archi_and_config["model_config"].modalities_specific_dim is not None:
            assert not outputs.one_latent_space
        else:
            assert outputs.one_latent_space

        # Test the shape of the shared embeddings
        embeddings = outputs.z
        assert isinstance(outputs, ModelOutput)
        assert embeddings.shape == (1, latent_dim)
        embeddings = model.encode(dataset[0], N=2).z
        assert embeddings.shape == (2, 1, latent_dim)
        embeddings = model.encode(dataset, cond_mod=["mod2"]).z
        assert embeddings.shape == (len(dataset), latent_dim)
        embeddings = model.encode(dataset, cond_mod="mod3", N=10).z
        assert embeddings.shape == (10, len(dataset), latent_dim)
        embeddings = model.encode(dataset, cond_mod=["mod2", "mod4"]).z
        assert embeddings.shape == (len(dataset), latent_dim)

        # Test that the encode function returns the private embeddings
        if model.multiple_latent_spaces:
            output = model.encode(dataset[0], N=2)
            assert not output.one_latent_space
            assert hasattr(output, "modalities_z")

        # Test the return_mean parameter
        for cond_mod in ["all", ["mod2", "mod3"]]:
            outputs = model.encode(
                dataset[:3], cond_mod=cond_mod, return_mean=True, N=5
            )
            assert outputs.z.shape == (5, 3, latent_dim)
            # Assert that the returned embeddings contains 3 times the mean
            assert torch.all(outputs.z[1:] == outputs.z[0])

    def test_predict(self, model, dataset):
        """Test the predict method of the MoPoE model.
        We check the shape of the reconstructions depending on the parameters.
        """
        # Test the shape of reconstruction
        Y = model.predict(dataset, cond_mod="mod2")
        assert isinstance(Y, ModelOutput)
        assert Y.mod1.shape == (len(dataset), 2)
        assert Y.mod2.shape == (len(dataset), 3)

        Y = model.predict(dataset, cond_mod="mod2", N=10)
        assert isinstance(Y, ModelOutput)
        assert Y.mod1.shape == (10, len(dataset), 2)
        assert Y.mod2.shape == (10, len(dataset), 3)

        Y = model.predict(dataset, cond_mod="mod2", N=10, flatten=True)
        assert isinstance(Y, ModelOutput)
        assert Y.mod1.shape == (len(dataset) * 10, 2)
        assert Y.mod2.shape == (len(dataset) * 10, 3)

    def test_random_mixture(self, model):
        """Test the model random_mixture_selection_component_selection on
        a simple example.
        """
        mus = torch.arange(3 * 2 * 4).reshape(3, 2, 4)
        log_vars = torch.arange(3 * 2 * 4).reshape(3, 2, 4)
        avail = torch.tensor([[1, 0], [0, 1], [0, 0]])

        mu_joint, _ = model.random_mixture_component_selection(mus, log_vars, avail)

        assert torch.all(mu_joint == torch.tensor([[0, 1, 2, 3], [12, 13, 14, 15]]))

    def test_backward_with_missing(self, model, dataset):
        """Check that the gradient with regard to missing modalities is null."""
        ### Check that the grad with regard to missing modalities is null
        if hasattr(dataset, "masks"):
            # Test that the gradients are null for the missing modalities
            output = model(dataset[3:], epoch=2)
            loss = output.loss
            loss.backward()
            for param in model.encoders["mod1"].parameters():
                assert torch.all(param.grad == 0)

            # Test that the gradient is not null when modalities are present
            output = model(dataset[:3], epoch=2)
            loss = output.loss
            loss.backward()
            for param in model.encoders["mod1"].parameters():
                assert not torch.all(param.grad == 0)

    @pytest.fixture
    def training_config(self, tmp_path_factory):
        """Create training config for test"""
        dir_path = tmp_path_factory.mktemp("dummy_folder")

        yield BaseTrainerConfig(
            num_epochs=3,
            steps_saving=2,
            learning_rate=1e-4,
            optimizer_cls="AdamW",
            optimizer_params={"betas": (0.91, 0.995)},
            output_dir=str(dir_path),
            no_cuda=True,
        )
        shutil.rmtree(dir_path)

    @pytest.fixture
    def trainer(self, model, training_config, dataset):
        """Create a trainer for testing the model"""
        trainer = BaseTrainer(
            model=model,
            train_dataset=dataset,
            eval_dataset=dataset,
            training_config=training_config,
        )

        trainer.prepare_training()

        return trainer

    @pytest.mark.slow
    def test_train_step(self, trainer):
        """Test the train step with the MoPoE model.
        Weights sould be updated.
        """
        start_model_state_dict = deepcopy(trainer.model.state_dict())
        start_optimizer = trainer.optimizer
        _ = trainer.train_step(epoch=1)

        step_1_model_state_dict = deepcopy(trainer.model.state_dict())

        # check that weights were updated
        assert not all(
            [
                torch.equal(start_model_state_dict[key], step_1_model_state_dict[key])
                for key in start_model_state_dict.keys()
            ]
        )
        assert trainer.optimizer == start_optimizer

    @pytest.mark.slow
    def test_eval_step(self, trainer):
        """Test eval step with the MoPoE model.
        Weights should not be updated.
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

    @pytest.mark.slow
    def test_main_train_loop(self, trainer):
        """Test the main train loop with MoPoE model.
        Weights should be updated.
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

    @pytest.mark.slow
    def test_checkpoint_saving(self, model, trainer, training_config):
        """Test checkpoint saving with the MoPoE model.
        Check that the model state dict and custom architectures are saved to folder.
        """
        dir_path = training_config.output_dir

        # Make a training step
        trainer.train_step(epoch=1)

        model = deepcopy(trainer.model)
        optimizer = deepcopy(trainer.optimizer)

        trainer.save_checkpoint(dir_path=dir_path, epoch=0, model=model)

        checkpoint_dir = os.path.join(dir_path, "checkpoint_epoch_0")

        assert os.path.isdir(checkpoint_dir)

        files_list = os.listdir(checkpoint_dir)

        assert set(["model.pt", "optimizer.pt", "training_config.json"]).issubset(
            set(files_list)
        )

        # check pickled custom architectures
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

        # check reload full model
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

    @pytest.mark.slow
    def test_checkpoint_saving_during_training(self, model, trainer, training_config):
        """Test the creation of checkpoints during training."""
        target_saving_epoch = training_config.steps_saving

        dir_path = training_config.output_dir

        model = deepcopy(trainer.model)

        trainer.train()

        training_dir = os.path.join(
            dir_path, f"MoPoE_training_{trainer._training_signature}"
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

    @pytest.mark.slow
    def test_final_model_saving(self, model, trainer, training_config):
        """Test the final model saving of MoPoE.
        We check that the model can be reloaded correctly with AutoModel.
        """
        dir_path = training_config.output_dir

        trainer.train()

        model = deepcopy(trainer._best_model)

        training_dir = os.path.join(
            dir_path, f"MoPoE_training_{trainer._training_signature}"
        )
        assert os.path.isdir(training_dir)

        final_dir = os.path.join(training_dir, "final_model")
        assert os.path.isdir(final_dir)

        files_list = os.listdir(final_dir)

        assert set(["model.pt", "model_config.json", "training_config.json"]).issubset(
            set(files_list)
        )

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

    def test_compute_nll(self, model, dataset):
        """Check the compute_joint_nll method of the MoPoE model."""
        if hasattr(dataset, "masks"):
            with pytest.raises(AttributeError):
                nll = model.compute_joint_nll(dataset, K=10, batch_size_K=6)
        else:
            nll = model.compute_joint_nll(dataset, K=10, batch_size_K=6)
            assert nll >= 0
            assert isinstance(nll, torch.Tensor)
            assert nll.size() == torch.Size([])

    def test_compute_joint_nll_from_subset_encoding(self, model, dataset):
        """Test the compute_joint_nll_from_subset_encoding of the MoPoE model."""
        if hasattr(dataset, "masks"):
            with pytest.raises(AttributeError):
                nll = model.compute_joint_nll_from_subset_encoding(
                    dataset, K=10, batch_size_K=6
                )
        else:
            nll = model._compute_joint_nll_from_subset_encoding(
                ["mod1", "mod2"], dataset, K=10, batch_size_K=6
            )
            assert nll >= 0
            assert isinstance(nll, torch.Tensor)
            assert nll.size() == torch.Size([])

    def test_compute_nll_paper(self, model, dataset):
        """Test the compute_joint_nll_paper method.
        We check that a positive value is computed.
        """
        if hasattr(dataset, "masks"):
            with pytest.raises(AttributeError):
                nll = model.compute_joint_nll_paper(dataset, K=10, batch_size_K=6)
        else:
            nll = model.compute_joint_nll_paper(dataset, K=10, batch_size_K=6)
            assert nll >= 0
            assert isinstance(nll, torch.Tensor)
            assert nll.size() == torch.Size([])
