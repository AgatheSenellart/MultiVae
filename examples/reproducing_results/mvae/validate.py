from multivae.data.datasets.celeba import CelebAttr
from multivae.data.datasets.mnist_labels import BinaryMnistLabels
from multivae.metrics import LikelihoodsEvaluator, LikelihoodsEvaluatorConfig
from multivae.models import AutoModel

model = AutoModel.load_from_hf_hub("asenella/reproduce_mvae_mnist_1", allow_pickle=True)

test_set = BinaryMnistLabels(data_path="../data", split="test", random_binarized=True)

ll_config = LikelihoodsEvaluatorConfig(batch_size=512, K=1000, batch_size_k=500)

ll_module = LikelihoodsEvaluator(model, test_set, eval_config=ll_config)

ll_module.eval()
