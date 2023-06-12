import numpy as np
import torch
from pythae.models.base.base_model import BaseDecoder, BaseEncoder
from pythae.models.base.base_utils import ModelOutput
from pythae.models.nn.benchmarks.utils import ResBlock
from torch import nn

from multivae.models.base.base_config import BaseAEConfig


class Flatten(torch.nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class Unflatten(torch.nn.Module):
    def __init__(self, ndims):
        super(Unflatten, self).__init__()
        self.ndims = ndims

    def forward(self, x):
        return x.view(x.size(0), *self.ndims)


class EncoderConvMMNIST(BaseEncoder):
    """
    Adopted from:
    https://www.cs.toronto.edu/~lczhang/360/lec/w05/autoencoder.html
    """

    def __init__(self, model_config: BaseAEConfig):
        super(EncoderConvMMNIST, self).__init__()
        self.latent_dim = model_config.latent_dim
        self.shared_encoder = nn.Sequential(  # input shape (3, 28, 28)
            nn.Conv2d(
                3, 32, kernel_size=3, stride=2, padding=1, bias=True
            ),  # -> (32, 14, 14)
            nn.ReLU(),
            nn.Conv2d(
                32, 64, kernel_size=3, stride=2, padding=1, bias=True
            ),  # -> (64, 7, 7)
            nn.ReLU(),
            nn.Conv2d(
                64, 128, kernel_size=3, stride=2, padding=1, bias=True
            ),  # -> (128, 4, 4)
            nn.ReLU(),
            # nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=True),  # -> (256, 2, 2)
            # nn.ReLU(),
            nn.Flatten(),
            nn.Linear(2048, self.latent_dim),  # -> (ndim_private + ndim_shared)
            nn.ReLU(),
        )

        # content branch
        self.class_mu = nn.Linear(self.latent_dim, self.latent_dim)
        self.class_logvar = nn.Linear(self.latent_dim, self.latent_dim)

    def forward(self, x):
        h = self.shared_encoder(x)
        return ModelOutput(
            embedding=self.class_mu(h), log_covariance=self.class_logvar(h)
        )


class EncoderConvMMNIST_adapted(BaseEncoder):
    """
    Adapt so that it works with DCCA and models with multiple latent spaces.
    """

    def __init__(self, model_config: BaseAEConfig):
        super(EncoderConvMMNIST_adapted, self).__init__()
        self.latent_dim = model_config.latent_dim
        self.style_dim = model_config.style_dim
        self.shared_encoder = nn.Sequential(  # input shape (3, 28, 28)
            nn.Conv2d(
                3, 32, kernel_size=3, stride=2, padding=1, bias=True
            ),  # -> (32, 14, 14)
            nn.ReLU(),
            nn.Conv2d(
                32, 64, kernel_size=3, stride=2, padding=1, bias=True
            ),  # -> (64, 7, 7)
            nn.ReLU(),
            nn.Conv2d(
                64, 128, kernel_size=3, stride=2, padding=1, bias=True
            ),  # -> (128, 4, 4)
            nn.ReLU(),
        )

        # content branch
        self.class_mu = nn.Conv2d(128, self.latent_dim, 4, 2, 0)
        self.class_logvar = nn.Conv2d(128, self.latent_dim, 4, 2, 0)

    def forward(self, x):
        h = self.shared_encoder(x)
        return ModelOutput(
            embedding=self.class_mu(h).squeeze(),
            log_covariance=self.class_logvar(h).squeeze(),
        )


class EncoderConvMMNIST_multilatents(BaseEncoder):
    """
    Adapt so that it works with multiple latent spaces models.
    """

    def __init__(self, model_config: BaseAEConfig):
        super(EncoderConvMMNIST_multilatents, self).__init__()
        self.latent_dim = model_config.latent_dim
        self.style_dim = model_config.style_dim
        self.encoder_class = nn.Sequential(  # input shape (3, 28, 28)
            nn.Conv2d(
                3, 32, kernel_size=3, stride=2, padding=1, bias=True
            ),  # -> (32, 14, 14)
            nn.ReLU(),
            nn.Conv2d(
                32, 64, kernel_size=3, stride=2, padding=1, bias=True
            ),  # -> (64, 7, 7)
            nn.ReLU(),
            nn.Conv2d(
                64, 128, kernel_size=3, stride=2, padding=1, bias=True
            ),  # -> (128, 4, 4)
            nn.ReLU(),
        )

        # content branch
        self.class_mu = nn.Conv2d(128, self.latent_dim, 4, 2, 0)
        self.class_logvar = nn.Conv2d(128, self.latent_dim, 4, 2, 0)

        if self.style_dim > 0:
            self.encoder_style = nn.Sequential(  # input shape (3, 28, 28)
                nn.Conv2d(
                    3, 32, kernel_size=3, stride=2, padding=1, bias=True
                ),  # -> (32, 14, 14)
                nn.ReLU(),
                nn.Conv2d(
                    32, 64, kernel_size=3, stride=2, padding=1, bias=True
                ),  # -> (64, 7, 7)
                nn.ReLU(),
                nn.Conv2d(
                    64, 128, kernel_size=3, stride=2, padding=1, bias=True
                ),  # -> (128, 4, 4)
                nn.ReLU(),
            )

            self.style_mu = nn.Conv2d(128, self.style_dim, 4, 2, 0)
            self.style_logvar = nn.Conv2d(128, self.style_dim, 4, 2, 0)

    def forward(self, x):
        output = ModelOutput()
        # content branch
        h_class = self.encoder_class(x)
        output["embedding"] = self.class_mu(h_class).squeeze()
        output["log_covariance"] = self.class_logvar(h_class).squeeze()

        if self.style_dim > 0:
            # style branch
            h_style = self.encoder_style(x)
            output["style_embedding"] = self.style_mu(h_style).squeeze()
            output["style_log_covariance"] = self.style_logvar(h_style).squeeze()

        return output


class DecoderConvMMNIST(BaseDecoder):
    """
    Adopted from:
    https://www.cs.toronto.edu/~lczhang/360/lec/w05/autoencoder.html
    """

    def __init__(self, model_config: BaseAEConfig):
        super(DecoderConvMMNIST, self).__init__()
        self.latent_dim = model_config.latent_dim
        self.decoder = nn.Sequential(
            nn.Linear(self.latent_dim, 2048),  # -> (2048)
            nn.ReLU(),
            Unflatten((128, 4, 4)),  # -> (128, 4, 4)
            nn.ConvTranspose2d(
                128,
                64,
                kernel_size=3,
                stride=2,
                padding=1,
            ),  # -> (128, 4, 4)
            nn.ReLU(),
            nn.ConvTranspose2d(
                64, 32, kernel_size=3, stride=2, padding=1, output_padding=1
            ),  # -> (32, 14, 14)
            nn.ReLU(),
            nn.ConvTranspose2d(
                32, 3, kernel_size=3, stride=2, padding=1, output_padding=1
            ),  # -> (3, 28, 28)
        )

    def forward(self, z):
        x_hat = self.decoder(z.view(-1, z.size(-1)))
        # x_hat = torch.sigmoid(x_hat)
        x_hat = x_hat.view(*z.size()[:-1], *x_hat.size()[1:])
        return ModelOutput(
            reconstruction=x_hat
        )  # NOTE: consider learning scale param, too


class Encoder_ResNet_VAE_MMNIST(BaseEncoder):
    """
    A ResNet encoder suited for MNIST and Variational Autoencoder-based models.

    It can be built as follows:

    .. code-block::

        >>> from pythae.models.nn.benchmarks.mnist import Encoder_ResNet_VAE_MNIST
        >>> from pythae.models import VAEConfig
        >>> model_config = VAEConfig(input_dim=(1, 28, 28), latent_dim=16)
        >>> encoder = Encoder_ResNet_VAE_MNIST(model_config)
        >>> encoder
        ... Encoder_ResNet_VAE_MNIST(
        ...   (layers): ModuleList(
        ...     (0): Sequential(
        ...       (0): Conv2d(1, 64, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))
        ...     )
        ...     (1): Sequential(
        ...       (0): Conv2d(64, 128, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))
        ...     )
        ...     (2): Sequential(
        ...       (0): Conv2d(128, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        ...     )
        ...     (3): Sequential(
        ...       (0): ResBlock(
        ...         (conv_block): Sequential(
        ...           (0): ReLU()
        ...           (1): Conv2d(128, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        ...           (2): ReLU()
        ...           (3): Conv2d(32, 128, kernel_size=(1, 1), stride=(1, 1))
        ...         )
        ...       )
        ...       (1): ResBlock(
        ...         (conv_block): Sequential(
        ...           (0): ReLU()
        ...           (1): Conv2d(128, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        ...           (2): ReLU()
        ...           (3): Conv2d(32, 128, kernel_size=(1, 1), stride=(1, 1))
        ...         )
        ...       )
        ...     )
        ...   )
        ...   (embedding): Linear(in_features=2048, out_features=16, bias=True)
        ...   (log_var): Linear(in_features=2048, out_features=16, bias=True)
        ... )


    and then passed to a :class:`pythae.models` instance

        >>> from pythae.models import VAE
        >>> model = VAE(model_config=model_config, encoder=encoder)
        >>> model.encoder == encoder
        ... True

    .. note::

        Please note that this encoder is only suitable for Autoencoder based models since it only
        outputs the embeddings of the input data under the key `embedding`.

        .. code-block::

            >>> import torch
            >>> input = torch.rand(2, 1, 28, 28)
            >>> out = encoder(input)
            >>> out.embedding.shape
            ... torch.Size([2, 16])

    """

    def __init__(self, args: BaseAEConfig):
        BaseEncoder.__init__(self)

        self.input_dim = (3, 28, 28)
        self.latent_dim = args.latent_dim
        self.n_channels = 3

        layers = nn.ModuleList()

        layers.append(nn.Sequential(nn.Conv2d(self.n_channels, 64, 4, 2, padding=1)))

        layers.append(nn.Sequential(nn.Conv2d(64, 128, 4, 2, padding=1)))

        layers.append(nn.Sequential(nn.Conv2d(128, 128, 3, 2, padding=1)))

        layers.append(
            nn.Sequential(
                ResBlock(in_channels=128, out_channels=32),
                ResBlock(in_channels=128, out_channels=32),
            )
        )

        self.layers = layers
        self.depth = len(layers)

        self.embedding = nn.Linear(128 * 4 * 4, args.latent_dim)
        self.log_var = nn.Linear(128 * 4 * 4, args.latent_dim)

    def forward(self, x: torch.Tensor):
        """Forward method

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            ModelOutput: An instance of ModelOutput containing the embeddings of the input data
            under the key `embedding`. Optional: The outputs of the layers specified in
            `output_layer_levels` arguments are available under the keys `embedding_layer_i` where
            i is the layer's level."""
        output = ModelOutput()

        max_depth = self.depth

        out = x

        for i in range(max_depth):
            out = self.layers[i](out)
            if i + 1 == self.depth:
                output["embedding"] = self.embedding(out.reshape(x.shape[0], -1))
                output["log_covariance"] = self.log_var(out.reshape(x.shape[0], -1))

        return output


class Decoder_ResNet_AE_MMNIST(BaseDecoder):
    """
    A ResNet decoder suited for MNIST and Autoencoder-based
    models.

    .. code-block::

        >>> from pythae.models.nn.benchmarks.mnist import Decoder_ResNet_AE_MNIST
        >>> from pythae.models import VAEConfig
        >>> model_config = VAEConfig(input_dim=(1, 28, 28), latent_dim=16)
        >>> decoder = Decoder_ResNet_AE_MNIST(model_config)
        >>> decoder
        ... Decoder_ResNet_AE_MNIST(
        ...   (layers): ModuleList(
        ...     (0): Linear(in_features=16, out_features=2048, bias=True)
        ...     (1): ConvTranspose2d(128, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        ...     (2): Sequential(
        ...       (0): ResBlock(
        ...         (conv_block): Sequential(
        ...           (0): ReLU()
        ...           (1): Conv2d(128, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        ...           (2): ReLU()
        ...           (3): Conv2d(32, 128, kernel_size=(1, 1), stride=(1, 1))
        ...         )
        ...       )
        ...       (1): ResBlock(
        ...         (conv_block): Sequential(
        ...           (0): ReLU()
        ...           (1): Conv2d(128, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        ...           (2): ReLU()
        ...           (3): Conv2d(32, 128, kernel_size=(1, 1), stride=(1, 1))
        ...         )
        ...       )
        ...       (2): ReLU()
        ...     )
        ...     (3): Sequential(
        ...       (0): ConvTranspose2d(128, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), output_padding=(1, 1))
        ...       (1): ReLU()
        ...     )
        ...     (4): Sequential(
        ...       (0): ConvTranspose2d(64, 1, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), output_padding=(1, 1))
        ...       (1): Sigmoid()
        ...     )
        ...   )
        ... )



    and then passed to a :class:`pythae.models` instance

        >>> from pythae.models import VAE
        >>> model = VAE(model_config=model_config, decoder=decoder)
        >>> model.decoder == decoder
        ... True

    .. note::

        Please note that this decoder is suitable for **all** models.

        .. code-block::

            >>> import torch
            >>> input = torch.randn(2, 16)
            >>> out = decoder(input)
            >>> out.reconstruction.shape
            ... torch.Size([2, 1, 28, 28])
    """

    def __init__(self, args: BaseAEConfig):
        BaseDecoder.__init__(self)

        self.input_dim = (3, 28, 28)
        self.latent_dim = args.latent_dim
        self.n_channels = 3

        layers = nn.ModuleList()

        layers.append(nn.Linear(args.latent_dim, 128 * 4 * 4))

        layers.append(nn.ConvTranspose2d(128, 128, 3, 2, padding=1))

        layers.append(
            nn.Sequential(
                ResBlock(in_channels=128, out_channels=32),
                ResBlock(in_channels=128, out_channels=32),
                nn.ReLU(),
            )
        )

        layers.append(
            nn.Sequential(
                nn.ConvTranspose2d(128, 64, 3, 2, padding=1, output_padding=1),
                nn.ReLU(),
            )
        )

        layers.append(
            nn.Sequential(
                nn.ConvTranspose2d(
                    64, self.n_channels, 3, 2, padding=1, output_padding=1
                ),
                nn.Sigmoid(),
            )
        )

        self.layers = layers
        self.depth = len(layers)

    def forward(self, z: torch.Tensor):
        """Forward method

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            ModelOutput: An instance of ModelOutput containing the reconstruction of the latent code
            under the key `reconstruction`. Optional: The outputs of the layers specified in
            `output_layer_levels` arguments are available under the keys `reconstruction_layer_i`
            where i is the layer's level.
        """
        output = ModelOutput()

        max_depth = self.depth

        out = z

        for i in range(max_depth):
            out = self.layers[i](out)

            if i == 0:
                output_shape = (np.prod(z.shape[:-1]),) + (128, 4, 4)
                out = out.reshape(*output_shape)

            if i + 1 == self.depth:
                output_shape = (*z.shape[:-1],) + (3, 28, 28)
                output["reconstruction"] = out.reshape(output_shape)

        return output
