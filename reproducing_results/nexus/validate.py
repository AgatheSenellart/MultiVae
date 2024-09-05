##############################################################
# In this file, we reproduce the experiments from the Nexus paper :
#      "Leveraging hierarchy in multimodal generative models for effective cross-modality inference" (Vasco et al 2022)
#
# We validate trained models using petrained classifiers


from multivae.metrics import CoherenceEvaluator, CoherenceEvaluatorConfig
from multivae.models.auto_model import AutoModel 
from multivae.data.datasets import MHD
from classifiers import *
import os


device = 'cuda' if torch.cuda.is_available() else 'cpu'

test_set = MHD('/home/asenella/scratch/data/MHD', split='test', modalities=['audio', 'trajectory', 'image', 'label'])



####  Load classifiers
classifiers_path = '/home/asenella/scratch/data/MHD/classifiers'

classifiers = dict(
    image = Image_Classifier(),
    audio = Sound_Classifier(),
    trajectory = Trajectory_Classifier(), 
    label = Label_Classifier()
)

state_dicts = dict(
    image = torch.load(os.path.join(classifiers_path, 'best_image_classifier_model.pth.tar'), map_location=device)['state_dict'],
    audio = torch.load(os.path.join(classifiers_path, 'best_sound_classifier_model.pth.tar'), map_location=device)['state_dict'],
    trajectory = torch.load(os.path.join(classifiers_path, 'best_trajectory_classifier_model.pth.tar'), map_location=device)['state_dict'],
)

for s in state_dicts:
    classifiers[s].load_state_dict(state_dicts[s])
    classifiers[s].eval()


# Load the model 
model_path = '/home/asenella/dev/multivae_package/dummy_output_dir/NEXUS_training_2024-08-28_16-58-08/final_model'
model = AutoModel.load_from_folder(model_path)


eval_config = CoherenceEvaluatorConfig(
    batch_size=64,
    wandb_path="multimodal_vaes/reproducing_nexus/yokuodyw",
    num_classes=10
)

eval_module = CoherenceEvaluator(model = model, 
                                 classifiers= classifiers,
                                 test_dataset=test_set,
                                 output=model_path,
                                 eval_config = eval_config
                                 )

eval_module.cross_coherences()