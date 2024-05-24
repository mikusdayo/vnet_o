import torch


class UNet3dParts:
    @staticmethod
    def build_layers(input_channels, num_classes, conv3d_transpose_paddings):
        return torch.nn.ModuleDict(
            {
                **UNet3dParts.__create_encoder_layers(input_channels),
                # The "3D U-Net: Learning Dense Volumetric Segmentation from Sparse Annotation" paper explicitly states
                # that it doesn't use bottleneck layers. This is where one would usually live, hence the name.
                "not_bottleneck": UNet3dParts.__build_conv_block(
                    in_channels=256, intermediate_out_channels=256, out_channels=512
                ),
                **UNet3dParts.__create_decoder_layers(),
                **UNet3dParts.__create_upsampling_layers(conv3d_transpose_paddings),
                "max_pool": torch.nn.MaxPool3d(
                    kernel_size=2, stride=2, return_indices=False
                ),
                "output_conv": torch.nn.Conv3d(
                    in_channels=64,
                    out_channels=num_classes,
                    kernel_size=3,
                    padding="same",
                ),
                "output_activation": torch.nn.Softmax(dim=1)
                if num_classes > 1
                else torch.nn.Sigmoid(),
            }
        )

    @staticmethod
    def __create_encoder_layers(input_channels):
        layers = {}
        for level in range(1, 4):
            layers[f"encoder_level_{level}"] = UNet3dParts.__build_conv_block(
                in_channels=input_channels if level == 1 else 64 * 2 ** (level - 2),
                intermediate_out_channels=32 * 2 ** (level - 1),
                out_channels=64 * 2 ** (level - 1),
            )
        return layers

    @staticmethod
    def __create_decoder_layers():
        layers = {}
        for level in range(1, 4):
            layers[f"decoder_level_{level}"] = UNet3dParts.__build_conv_block(
                in_channels=int(1.5 * 128 * 2 ** (level - 1)),
                intermediate_out_channels=64 * 2 ** (level - 1),
                out_channels=64 * 2 ** (level - 1),
            )
        return layers

    @staticmethod
    def __create_upsampling_layers(conv3d_transpose_paddings):
        layers = {}
        for level in range(2, 5):
            layers[f"upsampling_level_{level}"] = UNet3dParts.__build_conv3d_transpose(
                128 * 2 ** (level - 2), output_padding=conv3d_transpose_paddings[level]
            )
        return layers

    @staticmethod
    def __build_conv_block(in_channels, intermediate_out_channels, out_channels):
        layers = [
            torch.nn.Conv3d(
                in_channels=in_channels,
                out_channels=intermediate_out_channels,
                kernel_size=3,
                padding="same",
            ),
            torch.nn.BatchNorm3d(num_features=intermediate_out_channels),
            torch.nn.ReLU(),
            torch.nn.Conv3d(
                in_channels=intermediate_out_channels,
                out_channels=out_channels,
                kernel_size=3,
                padding="same",
            ),
            torch.nn.BatchNorm3d(num_features=out_channels),
            torch.nn.ReLU(),
        ]
        return torch.nn.Sequential(*layers)

    @staticmethod
    def __build_conv3d_transpose(channels, output_padding):
        return torch.nn.ConvTranspose3d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=2,
            stride=2,
            output_padding=output_padding,
        )
