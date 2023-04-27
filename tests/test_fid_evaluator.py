import pytest
import torch
from pythae.models.base.base_utils import ModelOutput

from multivae.data.datasets import MultimodalBaseDataset
from multivae.metrics import FIDEvaluator, FIDEvaluatorConfig
from multivae.models import MVTCAE, MVTCAEConfig
from multivae.models.nn.default_architectures import Encoder_VAE_MLP


class Test:
    
    @pytest.fixture(
        params=['custom_config','default_config']
    )
    def config(self, request):
        config = FIDEvaluatorConfig(batch_size=64,resize=False,
                                    inception_weights_path='./fids.pt')
        if request.param == 'custom_config':
            return config
        else :
            return FIDEvaluatorConfig(batch_size=64)
        
    @pytest.fixture(
        params=[None,lambda x:x]
    )
    def fid_model(self,config, request):
        model_config = MVTCAEConfig(n_modalities=2,
                                    input_dims={'m0': (3,32,32),'m1' :(1,28,28)},
                                    )
        model = MVTCAE(model_config)
        test_dataset = MultimodalBaseDataset(data = {
            'm0' : torch.randn((128,3,32,32)),
            'm1' : torch.randn((128,1,28,28)),

        },
                                             labels=torch.ones((1024,)))
        
        return FIDEvaluator(model=model,
                            test_dataset=test_dataset,
                                output=None,
                                eval_config=config,
                                custom_encoder=None,
                                transform=request.param) # Add test with custom encoder
        
        
    
    def test(self,fid_model,config):
        assert(fid_model.n_data) == 128
        assert len(fid_model.test_loader) > 0
        assert fid_model.batch_size == config.batch_size
        output = fid_model.eval()
        assert isinstance(output, ModelOutput)
        # assert hasattr(output, 'fd_m0')
        # assert hasattr(output, 'fd_m1')

        
        
        