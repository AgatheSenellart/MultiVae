from multivae.data.datasets.celeba import CelebAttr
from multivae.metrics import LikelihoodsEvaluator, LikelihoodsEvaluatorConfig
from multivae.models import AutoModel

model = AutoModel.load_from_hf_hub('asenella/reproducing_mvae', allow_pickle=True)

test_set = CelebAttr('~/scratch/data', split='test')

ll_config = LikelihoodsEvaluatorConfig(
    batch_size=512,
    K=100,
    batch_size_k=100
)

ll_module = LikelihoodsEvaluator(model,test_set,eval_config=ll_config)

ll_module.eval()