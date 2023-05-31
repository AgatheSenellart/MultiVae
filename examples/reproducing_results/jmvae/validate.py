import argparse

from multivae.data.datasets.mnist_labels import MnistLabels
from multivae.metrics import LikelihoodsEvaluator, LikelihoodsEvaluatorConfig
from multivae.models import AutoModel

model = AutoModel.load_from_hf_hub("asenella/reproduce_jmvae_seed_1", allow_pickle=True)

test_set = MnistLabels(data_path="../../../data", split="test")

ll_config = LikelihoodsEvaluatorConfig(
    K=1000,
    unified_implementation=False,
    wandb_path="multimodal_vaes/reproduce_jmvae/bizdz3q0",
)

ll_module = LikelihoodsEvaluator(model, test_set, eval_config=ll_config)

ll_module.eval()
