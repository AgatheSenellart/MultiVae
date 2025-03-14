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
from multivae.models.base import BaseDecoder, BaseEncoder, ModelOutput
from multivae.models.mhvae import MHVAE, MHVAEConfig
from multivae.models.nn.default_architectures import ModelOutput
from multivae.trainers import BaseTrainer, BaseTrainerConfig

# Architectures for testing


class my_input_encoder(BaseEncoder):

    def __init__(self):
        super().__init__()

        self.conv0 = nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=True)
        self.act_1 = nn.SiLU()

    def forward(self, x):

        x = self.conv0(x)
        x = self.act_1(x)

        return ModelOutput(embedding=x)


class bu_2(BaseEncoder):

    def __init__(self, inchannels, outchannels, latent_dim):
        super().__init__()

        self.network = nn.Sequential(
            nn.Conv2d(
                in_channels=inchannels,
                out_channels=outchannels,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=True,
            ),
            nn.SiLU(),
            nn.Flatten(),
            nn.Linear(2048, 512),
            nn.ReLU(),
        )

        self.mu = nn.Linear(512, latent_dim)
        self.log_var = nn.Linear(512, latent_dim)

    def forward(self, x):
        h = self.network(x)
        return ModelOutput(embedding=self.mu(h), log_covariance=self.log_var(h))


# Defininin top-down blocks and decoder


class td_2(nn.Module):

    def __init__(self, latent_dim):
        super().__init__()

        self.linear = nn.Sequential(nn.Linear(latent_dim, 2048), nn.ReLU())
        self.convs = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, bias=True),
            nn.SiLU(),
        )

    def forward(self, x):
        h = self.linear(x)
        h = h.view(h.shape[0], 128, 4, 4)
        return self.convs(h)


class td_1(nn.Module):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.network = nn.Sequential(
            nn.ConvTranspose2d(
                64, 32, kernel_size=3, stride=2, padding=1, output_padding=1, bias=True
            ),
            nn.SiLU(),
        )

    def forward(self, x):
        return self.network(x)


class bu_1(nn.Module):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.network = nn.Sequential(
            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=True,
            ),
            nn.SiLU(),
        )

    def forward(self, x):
        return self.network(x)


class add_bu(nn.Module):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.network = nn.Sequential(
            nn.Conv2d(
                in_channels=64,
                out_channels=64,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=True,
            ),
            nn.SiLU(),
        )

    def forward(self, x):
        return self.network(x)


class my_input_decoder(BaseDecoder):

    def __init__(self):
        super().__init__()

        self.network = nn.Sequential(
            nn.ConvTranspose2d(32, 3, 3, 2, 1, output_padding=1), nn.Sigmoid()
        )

    def forward(self, x):
        return ModelOutput(reconstruction=self.network(x))


# Defining prior blocks and posterior blocks


class prior_block(BaseEncoder):

    def __init__(self, n_channels, wn=False):
        super().__init__()
        if wn:
            self.mu = nn.utils.parametrizations.weight_norm(
                nn.Conv2d(n_channels, n_channels, 1, 1, 0)
            )
            self.logvar = nn.utils.parametrizations.weight_norm(
                nn.Conv2d(n_channels, n_channels, 1, 1, 0)
            )
        else:
            self.mu = nn.Conv2d(n_channels, n_channels, 1, 1, 0)
            self.logvar = nn.Conv2d(n_channels, n_channels, 1, 1, 0)

    def forward(self, x):
        return ModelOutput(embedding=self.mu(x), log_covariance=self.logvar(x))


class posterior_block(BaseEncoder):

    def __init__(self, n_channels_before_concat, wn=False):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(
                2 * n_channels_before_concat,
                n_channels_before_concat,
                3,
                1,
                1,
                bias=True,
            ),
            nn.SiLU(),
        )
        if wn:
            self.mu = nn.utils.parametrizations.weight_norm(
                nn.Conv2d(n_channels_before_concat, n_channels_before_concat, 1, 1, 0)
            )
            self.logvar = nn.utils.parametrizations.weight_norm(
                nn.Conv2d(n_channels_before_concat, n_channels_before_concat, 1, 1, 0)
            )
        else:
            self.mu = nn.Conv2d(
                n_channels_before_concat, n_channels_before_concat, 1, 1, 0
            )
            self.logvar = nn.Conv2d(
                n_channels_before_concat, n_channels_before_concat, 1, 1, 0
            )

    def forward(self, x):
        h = self.network(x)
        return ModelOutput(embedding=self.mu(h), log_covariance=self.logvar(h))


class Test_MHVAE:

    @fixture(params=["complete", "incomplete"])
    def dataset(self, request):
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

        return MHVAEConfig(
            n_modalities=2,
            latent_dim=request.param[3],
            decoders_dist=dict(m0=request.param[0][0], m1=request.param[0][1]),
            n_latent=request.param[1],
            beta=request.param[2],
        )

    @fixture(params=[True, False])
    def wn(self, request):
        return request.param

    @fixture(params=[True, False])
    def architectures(self, model_config, wn, request):

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

    @fixture
    def model(self, model_config, architectures):
        return MHVAE(model_config=model_config, **architectures)

    def test_sanity_check_bottom_up(self, model):
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

        wrong_bottom_up = deepcopy(model.bottom_up_blocks)
        wrong_bottom_up["m0"][-1] = wrong_bottom_up["m0"][-2]
        with pytest.raises(AttributeError):
            model.sanity_check_bottom_up(model.encoders, wrong_bottom_up)

        return

    def test_sanity_check_top_down(self, model):
        wrong_top_bottom = deepcopy(model.top_down_blocks)
        wrong_top_bottom = wrong_top_bottom[:-1]
        with pytest.raises(AttributeError):

            model.sanity_check_top_down_blocks(wrong_top_bottom)
        return

    def test_check_and_set_posterior_blocks(self, model):

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
        with pytest.raises(TypeError):
            archi = deepcopy(architectures)
            archi.pop("encoders")
            model = MHVAE(model_config=model_config, **archi)
        with pytest.raises(TypeError):
            architectures.pop("decoders")
            model = MHVAE(model_config=model_config, **architectures)

    def test_forward(self, model, dataset):

        samples = dataset[:10]
        output = model(samples)

        assert isinstance(output, ModelOutput)
        assert hasattr(output, "loss")
        assert isinstance(output.loss, torch.Tensor)

        return

    def test_encode(self, dataset, model):

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

        samples = dataset[:10]

        embeddings = model.encode(samples)
        output = model.decode(embeddings)

        assert isinstance(output, ModelOutput)
        assert hasattr(output, "m0")
        assert output.m0.shape == samples.data["m0"].shape

        return

    def test_predict(self, model, dataset):

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

        return BaseTrainer(
            model,
            train_dataset=dataset,
            eval_dataset=dataset,
            training_config=trainer_config,
        )

    @mark.slow
    def test_train_step(self, trainer):
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
        dir_path = trainer_config.output_dir

        # Make a training step, save the model and reload it
        step_1_loss = trainer.train_step(epoch=1)

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

            assert type(model_rec.encoders.cpu()) == type(model.encoders.cpu())
            assert type(model_rec.decoders.cpu()) == type(model.decoders.cpu())

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
        dir_path = trainer_config.output_dir

        trainer.train()

        model = deepcopy(trainer._best_model)

        training_dir = os.path.join(
            dir_path, f"MHVAE_training_{trainer._training_signature}"
        )
        assert os.path.isdir(training_dir)

        final_dir = os.path.join(training_dir, f"final_model")
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

            assert type(model_rec.encoders.cpu()) == type(model.encoders.cpu())
            assert type(model_rec.decoders.cpu()) == type(model.decoders.cpu())
