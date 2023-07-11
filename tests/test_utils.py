from multivae.models.mmvaePlus.utils import split_inputs, MultimodalBaseDataset, IncompleteDataset, DatasetOutput
import pytest
import torch
import numpy as np

class Test_split:
     

    @pytest.fixture(params=[True, False])
    def dataset(self, request):
        
        data = dict(
            m0 = torch.from_numpy(np.arange(25*3).reshape(25,3)),
            m1 = torch.from_numpy(np.arange(25*3*32*32).reshape(25,3,32,32))
        )
        
        masks = dict(
            m0 = torch.ones(25),
            m1 = torch.zeros(25)
        )
        if request.param:
            return MultimodalBaseDataset(
                data = data
                
            )
        else:
            return IncompleteDataset(
                data=data,
                masks=masks
            )
            
    
    def test(self, dataset):
        
        split = split_inputs(dataset,6)
        
        assert type(split) == list
        assert isinstance(split[0], DatasetOutput)
        
        assert hasattr(split[0], 'masks') or not hasattr(dataset, 'masks')
        
        assert len(split) == 7
        
        assert len(split[-1].data['m0']) ==1 
        
    def test(self, dataset):
        
        split = split_inputs(dataset,1)
        
        assert type(split) == list
        assert isinstance(split[0], DatasetOutput)
        
        assert hasattr(split[0], 'masks') or not hasattr(dataset, 'masks')
        
        assert len(split) == 1
        
        assert len(split[-1].data['m0']) == 25
        
        