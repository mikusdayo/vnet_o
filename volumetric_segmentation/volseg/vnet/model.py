import torch

from volseg.model.plottable_model import PlottableModel
from volseg.utils.padding import calculate_required_paddings
from volseg.vnet.parts import VNetParts


class VNet(PlottableModel):
    def __init__(self, num_classes, image_dimensions=(1, 64, 128, 128)):
        """
        :param image_dimensions: (channels, depth, height, width) or ImageDimensionsWrapper
        :param num_classes: number of classes to segment (e.g. liver, pancreas, lung...)
        """
        super().__init__(image_dimensions=image_dimensions)
        self.num_classes = num_classes
        conv3d_transpose_paddings = calculate_required_paddings(
            *self.image_dimensions.get_dhw(), num_levels=5
        )
        self.layers = VNetParts.build_layers(
            self.image_dimensions.channels, num_classes, conv3d_transpose_paddings
        )

    def forward(self, x):
        encoder_level_outputs, bottom_level_input = self.__encode(x)

        bottom_level_conv_block_output = self.layers["bottom_level"](bottom_level_input)
        bottom_level_output = bottom_level_conv_block_output + bottom_level_input
        decoder_output = self.__decode(encoder_level_outputs, bottom_level_output)
        decoder_output_adjusted_channels = self.layers["output_channels_adjust"](
            decoder_output
        )
        return self.layers["output_activation"](decoder_output_adjusted_channels)

    def __encode(self, x):
        layer_input = self.layers["input_channels_adjust"](x)
        level_outputs = {}
        for level in range(1, 5):
            encoder_conv_block_output = self.layers[f"encoder_level_{level}"](
                layer_input
            )
            encoder_level_output = encoder_conv_block_output + layer_input
            level_outputs[level] = encoder_level_output
            encoder_level_downsampled = self.layers[f"downsampling_level_{level}"](
                encoder_level_output
            )
            layer_input = encoder_level_downsampled
        return level_outputs, layer_input

    def __decode(self, encoder_level_outputs, bottom_level_output):
        upsampled = self.layers[f"upsampling_bottom_level"](bottom_level_output)
        for level in range(4, 0, -1):
            encoder_output = encoder_level_outputs[level]
            block_input = torch.concat((upsampled, encoder_output), axis=1)
            block_output = self.layers[f"decoder_level_{level}"](block_input)
            block_output_residual = upsampled + block_output
            if level > 1:
                upsampled = self.layers[f"upsampling_level_{level}"](
                    block_output_residual
                )
        # noinspection PyUnboundLocalVariable
        return block_output
