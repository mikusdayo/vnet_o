import torch


class VNetParts:
    @staticmethod
    def build_layers(input_channels, num_classes, conv3d_transpose_paddings):
        return torch.nn.ModuleDict(
            {
                **VNetParts.__create_channels_adjusters(input_channels, num_classes),
                **VNetParts.__create_encoder_layers(),
                **VNetParts.__create_decoder_layers(),
                **VNetParts.__create_downsampling_layers(),
                **VNetParts.__create_upsampling_layers(conv3d_transpose_paddings),
                "bottom_level": VNetParts.__build_conv_block(
                    convolutions_count=3, channels=256
                ),
                "output_activation": torch.nn.Softmax(dim=1)
                if num_classes > 1
                else torch.nn.Sigmoid(),
            }
        )

    @staticmethod
    def __create_encoder_layers():
        layers = {}
        for level in range(1, 5):
            layers[f"encoder_level_{level}"] = VNetParts.__build_conv_block(
                convolutions_count=min(level, 3),
                channels=16 * 2 ** (level - 1),
            )
        return layers

    @staticmethod
    def __create_downsampling_layers():
        layers = {}
        for level in range(1, 5):
            layers[
                f"downsampling_level_{level}"
            ] = VNetParts.__build_downsampling_layer(
                in_channels=16 * 2 ** (level - 1), out_channels=32 * 2 ** (level - 1)
            )
        return layers

    @staticmethod
    def __create_decoder_layers():
        layers = {}
        for level in range(1, 5):
            layers[f"decoder_level_{level}"] = VNetParts.__build_conv_block(
                convolutions_count=min(level, 3),
                in_channels=int(1.5 * 32 * 2 ** (level - 1)),
                channels=32 * 2 ** (level - 1),
            )
        return layers

    @staticmethod
    def __create_upsampling_layers(conv3d_transpose_paddings):
        layers = {}
        for level in range(2, 5):
            layers[f"upsampling_level_{level}"] = VNetParts.__build_conv3d_transpose(
                in_channels=64 * 2 ** (level - 2),
                out_channels=32 * 2 ** (level - 2),
                output_padding=conv3d_transpose_paddings[level],
            )
        layers["upsampling_bottom_level"] = VNetParts.__build_conv3d_transpose(
            in_channels=256,
            out_channels=256,
            output_padding=conv3d_transpose_paddings[5],
        )
        return layers

    @staticmethod
    def __create_channels_adjusters(input_channels, num_classes):
        return {
            "input_channels_adjust": torch.nn.Conv3d(
                in_channels=input_channels,
                out_channels=16,
                kernel_size=1,  # This 1x1x1 convolution adjusts channels to allow use in residual block
                stride=1,
            ),
            "output_channels_adjust": torch.nn.Conv3d(
                in_channels=32,
                out_channels=num_classes,
                kernel_size=1,  # This 1x1x1 convolution adjusts channels to return mask for all output classes
                stride=1,
            ),
        }

    @staticmethod
    def __build_downsampling_layer(in_channels, out_channels):
        return torch.nn.Conv3d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=2,
            stride=2,
        )

    @staticmethod
    def __build_conv_block(convolutions_count, channels, in_channels=None):
        if in_channels is None:
            in_channels = channels

        def build_conv_block_helper(is_first_in_sequence):
            return [
                torch.nn.Conv3d(
                    in_channels=in_channels if is_first_in_sequence else channels,
                    out_channels=channels,
                    kernel_size=5,
                    padding="same",
                ),
                torch.nn.PReLU(),
            ]

        layers = [
            layer
            for i in range(convolutions_count)
            for layer in build_conv_block_helper(is_first_in_sequence=i == 0)
        ]
        return torch.nn.Sequential(*layers)

    @staticmethod
    def __build_conv3d_transpose(in_channels, out_channels, output_padding):
        return torch.nn.ConvTranspose3d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=2,
            stride=2,
            output_padding=output_padding,
        )
