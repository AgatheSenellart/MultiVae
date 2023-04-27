from multivae.data.datasets.mnist_labels import BinaryMnistLabels
from multivae.metrics import LikelihoodsEvaluator, LikelihoodsEvaluatorConfig
from multivae.models import AutoModel

model = AutoModel.load_from_hf_hub('asenella/reproduce_jmvae', allow_pickle=True)

test_set = BinaryMnistLabels(data_path='../../../data', split='test')

ll_config = LikelihoodsEvaluatorConfig(
    K=1000
)

ll_module = LikelihoodsEvaluator(model,test_set,eval_config=ll_config)

ll_module.eval()