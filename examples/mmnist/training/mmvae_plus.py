from config2 import *

from multivae.models import MMVAEPlus, MMVAEPlusConfig

parser = argparse.ArgumentParser()
parser.add_argument("--param_file", type=str)
args = parser.parse_args()

with open(args.param_file, "r") as fp:
    info = json.load(fp)
args = argparse.Namespace(**info)

train_data = MMNISTDataset(
    data_path="~/scratch/data",
    split="train",
    missing_ratio=args.missing_ratio,
    keep_incomplete=args.keep_incomplete,
)

test_data = MMNISTDataset(data_path="~/scratch/data", split="test")

train_data, eval_data = random_split(
    train_data, [0.9, 0.1], generator=torch.Generator().manual_seed(42)
)

model_config = MMVAEPlusConfig(
    **base_config,
    K=1,
    prior_and_posterior_dist='laplace_with_softmax',
    learn_shared_prior=False,
    learn_modality_prior=True,
    beta=2.5,
    modalities_specific_dim=32,
    reconstruction_option="joint_prior",
)
model_config.latent_dim = 32

##### Architectures #####
from multivae.models.base import BaseEncoder, BaseDecoder, ModelOutput

def actvn(x):
    out = torch.nn.functional.leaky_relu(x, 2e-1)
    return out


class ResnetBlock(nn.Module):
    def __init__(self, fin, fout, fhidden=None, is_bias=True):
        super().__init__()
        # Attributes
        self.is_bias = is_bias
        self.learned_shortcut = fin != fout
        self.fin = fin
        self.fout = fout
        if fhidden is None:
            self.fhidden = min(fin, fout)
        else:
            self.fhidden = fhidden

        # Submodules
        self.conv_0 = nn.Conv2d(self.fin, self.fhidden, 3, stride=1, padding=1)
        self.conv_1 = nn.Conv2d(
            self.fhidden, self.fout, 3, stride=1, padding=1, bias=is_bias
        )
        if self.learned_shortcut:
            self.conv_s = nn.Conv2d(
                self.fin, self.fout, 1, stride=1, padding=0, bias=False
            )

    def forward(self, x):
        x_s = self._shortcut(x)
        dx = self.conv_0(actvn(x))
        dx = self.conv_1(actvn(dx))
        out = x_s + 0.1 * dx

        return out

    def _shortcut(self, x):
        if self.learned_shortcut:
            x_s = self.conv_s(x)
        else:
            x_s = x
        return x_s


# Classes
class Enc(BaseEncoder):
    """Generate latent parameters for SVHN image data."""

    def __init__(self, ndim_w, ndim_u):
        super().__init__()
        self.latent_dim = ndim_u
        s0 = self.s0 = 7  # kwargs['s0']
        nf = self.nf = 64  # nfilter
        nf_max = self.nf_max = 1024  # nfilter_max
        size = 28

        # Submodules
        nlayers = int(np.log2(size / s0))
        self.nf0 = min(nf_max, nf * 2**nlayers)

        blocks_w = [ResnetBlock(nf, nf)]

        blocks_u = [ResnetBlock(nf, nf)]

        for i in range(nlayers):
            nf0 = min(nf * 2**i, nf_max)
            nf1 = min(nf * 2 ** (i + 1), nf_max)
            blocks_w += [
                nn.AvgPool2d(3, stride=2, padding=1),
                ResnetBlock(nf0, nf1),
            ]
            blocks_u += [
                nn.AvgPool2d(3, stride=2, padding=1),
                ResnetBlock(nf0, nf1),
            ]

        self.conv_img_w = nn.Conv2d(3, 1 * nf, 3, padding=1)
        self.resnet_w = nn.Sequential(*blocks_w)
        self.fc_mu_w = nn.Linear(self.nf0 * s0 * s0, ndim_w)
        self.fc_lv_w = nn.Linear(self.nf0 * s0 * s0, ndim_w)

        self.conv_img_u = nn.Conv2d(3, 1 * nf, 3, padding=1)
        self.resnet_u = nn.Sequential(*blocks_u)
        self.fc_mu_u = nn.Linear(self.nf0 * s0 * s0, ndim_u)
        self.fc_lv_u = nn.Linear(self.nf0 * s0 * s0, ndim_u)

    def forward(self, x):
        # batch_size = x.size(0)
        out_w = self.conv_img_w(x)
        out_w = self.resnet_w(out_w)
        out_w = out_w.view(out_w.size()[0], self.nf0 * self.s0 * self.s0)
        lv_w = self.fc_lv_w(out_w)

        out_u = self.conv_img_u(x)
        out_u = self.resnet_u(out_u)
        out_u = out_u.view(out_u.size()[0], self.nf0 * self.s0 * self.s0)
        lv_u = self.fc_lv_u(out_u)

        output = ModelOutput(
            embedding=self.fc_mu_u(out_u),
            style_embedding=self.fc_mu_w(out_w),
            log_covariance=lv_u,
            style_log_covariance=lv_w,
        )

        return output


class Dec(BaseDecoder):
    """Generate a SVHN image given a sample from the latent space."""

    def __init__(self, ndim):
        super().__init__()

        # NOTE: I've set below variables according to Kieran's suggestions
        s0 = self.s0 = 7  # kwargs['s0']
        nf = self.nf = 64  # nfilter
        nf_max = self.nf_max = 512  # nfilter_max
        size = 28

        # Submodules
        nlayers = int(np.log2(size / s0))
        self.nf0 = min(nf_max, nf * 2**nlayers)

        self.fc = nn.Linear(ndim, self.nf0 * s0 * s0)

        blocks = []
        for i in range(nlayers):
            nf0 = min(nf * 2 ** (nlayers - i), nf_max)
            nf1 = min(nf * 2 ** (nlayers - i - 1), nf_max)
            blocks += [ResnetBlock(nf0, nf1), nn.Upsample(scale_factor=2)]

        blocks += [
            ResnetBlock(nf, nf),
        ]

        self.resnet = nn.Sequential(*blocks)
        self.conv_img = nn.Conv2d(nf, 3, 3, padding=1)

    def forward(self, z):
        out = self.fc(z).view(-1, self.nf0, self.s0, self.s0)
        out = self.resnet(out)
        out = self.conv_img(actvn(out))

        if len(z.size()) == 2:
            out = out.view(*z.size()[:1], *out.size()[1:])
        else:
            out = out.view(*z.size()[:2], *out.size()[1:])

        # consider also predicting the length scale
        return ModelOutput(reconstruction=out)


encoders = {
    m: Enc(model_config.modalities_specific_dim, ndim_u=model_config.latent_dim)
    for m in modalities
}
decoders = {
    m: Dec(model_config.latent_dim + model_config.modalities_specific_dim)
    for m in modalities
}

model = MMVAEPlus(model_config, encoders=encoders, decoders=decoders)

trainer_config = BaseTrainerConfig(
    **base_training_config,
    seed=args.seed,
    output_dir=f"compare_on_mmnist/{config_name}/{model.model_name}/seed_{args.seed}/missing_ratio_{args.missing_ratio}/K_{model.K}",
)
trainer_config.per_device_train_batch_size = 32
trainer_config.num_epochs = 150 if model.K==1 else 50 # enough for this model to reach convergence

# Set up callbacks
wandb_cb = WandbCallback()
wandb_cb.setup(trainer_config, model_config, project_name=wandb_project)
wandb_cb.run.config.update(args.__dict__)

callbacks = [TrainingCallback(), ProgressBarCallback(), wandb_cb]

trainer = BaseTrainer(
    model,
    train_dataset=train_data,
    eval_dataset=eval_data,
    training_config=trainer_config,
    callbacks=callbacks,
)
trainer.train()

model = trainer._best_model
save_model(model, args)
##################################################################################################################################
# validate the model #############################################################################################################
##################################################################################################################################

eval_model(model, trainer.training_dir, test_data, wandb_cb.run.path)
