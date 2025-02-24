import numpy as np
import pytest
import torch
from encoders import Encoder_test, Encoder_test_multilatents

from multivae.data.datasets.base import IncompleteDataset, MultimodalBaseDataset
from multivae.models.base.base_config import BaseAEConfig
from multivae.models import MVTCAE, MVTCAEConfig, MMVAEPlus, MMVAEPlusConfig
from multivae.models.nn.default_architectures import Decoder_AE_MLP, ModelOutput

from multivae.samplers.maf_sampler import MAFSampler, MAFSamplerConfig
import os

class Test_MAFSampler:

    @pytest.fixture
    def dataset(self):
        # Create simple small dataset
        data = dict(
            mod1=torch.Tensor([[1.0, 2.0], [4.0, 5.0]]),
            mod2=torch.Tensor([[67.1, 2.3, 3.0], [1.3, 2.0, 3.0]]),
            mod3=torch.Tensor([[37, 2, 4, 1], [8, 9, 7, 0]]),
            mod4=torch.Tensor([[37, 2, 4, 1], [8, 9, 7, 0]]),
        )
        labels = np.array([0, 1, 0, 0])
        dataset = MultimodalBaseDataset(data, labels)

        return dataset

    @pytest.fixture(params=[True, False])
    def one_latent_space(self, request):
        return request.param

    @pytest.fixture
    def archi_and_config(self,beta, one_latent_space):
        if one_latent_space:
            # Create an instance of mvae model
            config1 = BaseAEConfig(input_dim=(2,), latent_dim=5)
            config2 = BaseAEConfig(input_dim=(3,), latent_dim=5)
            config3 = BaseAEConfig(input_dim=(4,), latent_dim=5)

            encoders = dict(
                mod1=Encoder_test(config1),
                mod2=Encoder_test(config2),
                mod3=Encoder_test(config3),
                mod4=Encoder_test(config3),
            )

            model_config = MVTCAEConfig(
                n_modalities=4,
                latent_dim=5,
                input_dims=dict(mod1=(2,), mod2=(3,), mod3=(4,), mod4=(4,)),
                beta=beta,
            )

            decoders = dict(
                mod1=Decoder_AE_MLP(config1),
                mod2=Decoder_AE_MLP(config2),
                mod3=Decoder_AE_MLP(config3),
                mod4=Decoder_AE_MLP(config3),
            )

        else:
            config1 = BaseAEConfig(input_dim=(2,), latent_dim=5, style_dim=3)
            config2 = BaseAEConfig(input_dim=(3,), latent_dim=5, style_dim=3)
            config3 = BaseAEConfig(input_dim=(4,), latent_dim=5, style_dim=3)

            encoders = dict(
                mod1=Encoder_test_multilatents(config1),
                mod2=Encoder_test_multilatents(config2),
                mod3=Encoder_test_multilatents(config3),
                mod4=Encoder_test_multilatents(config3),
            )
            model_config = MMVAEPlusConfig(
                n_modalities=4,
                latent_dim=5,
                input_dims=dict(mod1=(2,), mod2=(3,), mod3=(4,), mod4=(4,)),
                beta=beta,
                modalities_specific_dim=3,
            )
            decoders = dict(
                mod1=Decoder_AE_MLP(BaseAEConfig(input_dim=(2,), latent_dim=9)),
                mod2=Decoder_AE_MLP(BaseAEConfig(input_dim=(3,), latent_dim=7)),
                mod3=Decoder_AE_MLP(BaseAEConfig(input_dim=(4,), latent_dim=8)),
                mod4=Decoder_AE_MLP(BaseAEConfig(input_dim=(4,), latent_dim=8)),
            )

        return dict(encoders=encoders, decoders=decoders, model_config=model_config)


    @pytest.fixture(params=[1.0, 1.5, 2.0])
    def beta(self,request):
        beta = request.param

        return beta


    @pytest.fixture(params=[True, False])
    def model(self,archi_and_config, one_latent_space, request):
        custom = request.param

        if one_latent_space:
            model_class = MVTCAE
        else:
            model_class = MMVAEPlus

        if custom:
            model = model_class(**archi_and_config)
        else:
            model = model_class(archi_and_config["model_config"])
        return model



    
    @pytest.fixture(params=[0, 1])
    def maf_sampler_config(self, request):
        if request.param == 0:
            return MAFSamplerConfig(n_made_blocks=2, n_hidden_in_made=4, hidden_size=64)
        else:
            return MAFSamplerConfig(
                n_made_blocks=1, n_hidden_in_made=1, hidden_size=16
            )

    
    def test_fit(self, maf_sampler_config, model, dataset, tmpdir):
        sampler = MAFSampler(model, maf_sampler_config)

        tmpdir.mkdir("dummy_folder")
        dir_path = os.path.join(tmpdir, "dummy_folder")

        # Test that trying to sample before fit raises an error:
        with pytest.raises(ArithmeticError):
            sampler.sample(100)
        
        with pytest.raises(AttributeError):
            sampler.load_flows_from_folder(dir_path)
        sampler.fit(dataset,eval_data=dataset)

        assert hasattr(sampler, "flows_models")

        assert sampler.is_fitted

        if sampler.model.multiple_latent_spaces:
            for m in sampler.model.encoders:
                assert m in sampler.flows_models.keys()

        # test sample
        output = sampler.sample(100)

        assert isinstance(output, ModelOutput)
        assert hasattr(output, "z")
        assert output.z.shape == (100, sampler.model.latent_dim)

        if sampler.model.multiple_latent_spaces:
            assert hasattr(output, "modalities_z")

        # test save
        
        sampler.save(dir_path)
        for m in sampler.maf_models:
            assert os.path.exists(os.path.join(dir_path, m))

        # Try reloading the config
        reload_config = MAFSamplerConfig.from_json_file(os.path.join(dir_path, 'sampler_config.json'))
        assert reload_config == sampler.sampler_config

        # Try reloading the flows
        reload_sampler = MAFSampler(model, maf_sampler_config)
        reload_sampler.load_flows_from_folder(dir_path)

        for m, model in reload_sampler.maf_models.items():
            assert all(
            [
                torch.equal(model.state_dict()[key].cpu(), sampler.maf_models[m].state_dict()[key].cpu())
                for key in model.state_dict().keys()
            ]
        )