from .transcoders import Conv2DTranscoder, VariationalTranscoder, GANTranscoder, GANDiscriminators
from .layers import GAN

def create_gan_model(**params):
    # Split generator & discriminator params
    g_params = {}
    d_params = {}
    for key, value in params.items():
        if key[:2] == "g_":
            # Generator param
            g_params[key[2:]] = value
        elif key[:2] == "d_":
            # Discriminator param
            d_params[key[2:]] = value
        else:
            # Shared param
            g_params[key] = value
            d_params[key] = value
    generator = GANTranscoder(**g_params)
    discriminator = GANDiscriminators(**d_params)
    gan = GAN(generator, discriminator)
    return gan.generator

input_shape = (261, 256, 1)
models = [
    {
        "label": "Convolutional Transcoder",
        "path": "./results/ConvolutionalTranscoder/model",
        "create_model": Conv2DTranscoder,
        "params": {
            "feature_shape": input_shape,
            "compression": 4,
            "kernel_size": 5,
            "conv_depth": 4,
            "input_chans_multiplier": 1,
            "skip_connection": True,
            "pooling_type": "average",
            "h_reg": 1.,
            "kl_reg": 0.,
        }
    },
    {
        "label": "Variational Transcoder",
        "path": "./results/VariationalTranscoder/model",
        "create_model": VariationalTranscoder,
        "params": {
            "feature_shape": input_shape,
            "compression": 4,
            "kernel_size": 5,
            "conv_depth": 4,
            "input_chans_multiplier": 1,
            "skip_connection": True,
            "pooling_type": "average",
            "h_reg": 1e-5,
            "kl_reg": 1e-12,
        }
    },
    # {
    #     "label": "GAN Transcoder with AdaIN",
    #     "path": "./results/GANTranscoderALT/model",
    #     "create_model": create_gan_model,
    #     "params": {
    #         "feature_shape": input_shape,
    #         "g_compression": 4,
    #         "g_kernel_size": 5,
    #         "g_conv_depth": 4,
    #         "g_input_chans_multiplier": 1,
    #         "g_skip_connection": True,
    #         "g_pooling_type": "average",
    #         "g_gan_reg": 0.02,
    #         "g_c_reg": 0.01,
    #         "g_s_reg": 0.01,
    #         "g_mode": "adain",
    #         "g_hidden_activation": "relu",
    #         "g_use_fake_style": False,
    #         "d_mlp_layers": 2,
    #         "d_conv_layers": 2,
    #         "d_conv_kernel_size": 3,
    #         "d_conv_pooling_size": 4,
    #         "d_conv_pooling_type": "max",
    #     }
    # },
    # {
    #     "label": "Variational GAN Transcoder with AdaIN",
    #     "path": "./results/GANTranscoder/model",
    #     "create_model": create_gan_model,
    #     "params": {
    #         "feature_shape": input_shape,
    #         "g_compression": 4,
    #         "g_kernel_size": 5,
    #         "g_conv_depth": 4,
    #         "g_input_chans_multiplier": 1,
    #         "g_skip_connection": True,
    #         "g_pooling_type": "average",
    #         "g_gan_reg": 0.02,
    #         "g_c_reg": 0.01,
    #         "g_s_reg": 0.01,
    #         "g_mode": "adain",
    #         "g_hidden_activation": "relu",
    #         "g_use_fake_style": True,
    #         "d_mlp_layers": 2,
    #         "d_conv_layers": 2,
    #         "d_conv_kernel_size": 3,
    #         "d_conv_pooling_size": 4,
    #         "d_conv_pooling_type": "max",
    #     }
    # },
    {
        "label": "GAN Transcoder",# with Learnable Style Code",
        "path": "./results/MUNITTranscoderALT/model",
        "create_model": create_gan_model,
        "params": {
            "feature_shape": input_shape,
            "g_compression": 4,
            "g_kernel_size": 5,
            "g_conv_depth": 4,
            "g_input_chans_multiplier": 1,
            "g_skip_connection": True,
            "g_pooling_type": "average",
            "g_gan_reg": 0.02,
            "g_c_reg": 0.01,
            "g_s_reg": 0.01,
            "g_use_fake_style": False,
            "g_is_munit": True,
            "g_style_dim": 8,
            "g_adain_momentum": 0.1,
            "g_adain_epsilon": 1e-5,
            "d_mlp_layers": 2,
            "d_conv_layers": 2,
            "d_conv_kernel_size": 3,
            "d_conv_pooling_size": 4,
            "d_conv_pooling_type": "max",
        }
    },
    {
        "label": "Variational GAN Transcoder",# with Learnable Style Code",
        "path": "./results/MUNITTranscoder/model",
        "create_model": create_gan_model,
        "params": {
            "feature_shape": input_shape,
            "g_compression": 4,
            "g_kernel_size": 5,
            "g_conv_depth": 4,
            "g_input_chans_multiplier": 1,
            "g_skip_connection": True,
            "g_pooling_type": "average",
            "g_gan_reg": 0.02,
            "g_c_reg": 0.01,
            "g_s_reg": 0.01,
            "g_use_fake_style": True,
            "g_is_munit": True,
            "g_style_dim": 8,
            "g_adain_momentum": 0.1,
            "g_adain_epsilon": 1e-5,
            "d_mlp_layers": 2,
            "d_conv_layers": 2,
            "d_conv_kernel_size": 3,
            "d_conv_pooling_size": 4,
            "d_conv_pooling_type": "max",
        }
    },
]