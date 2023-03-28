from pytest import fixture
from multivae.data.datasets.celeba_masks import CelebAMasks
from PIL.Image import Image
from torch import Tensor, Size

class Test_dataset:
    
    @fixture
    def dataset(self):
        return CelebAMasks()
    
    def test_get_item(self,dataset):
        item,mask = dataset[0]
        assert isinstance(item,Tensor)
        assert item.size() == Size([3,512,512])
        
        assert isinstance(mask,Tensor)
        assert mask.size() == Size([3,512,512])
        