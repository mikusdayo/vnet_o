import torch

from volseg.model.plottable_model import PlottableModel
from volseg.unet_3d.parts import UNet3dParts
from volseg.utils.padding import calculate_required_paddings


class UNet3d(PlottableModel):
    def __init__(self, num_classes, image_dimensions=(3, 116, 132, 132)):
        """
        :param image_dimensions: (channels, depth, height, width) or ImageDimensionsWrapper
        :param num_classes: number of classes to segment (e.g. liver, pancreas, lung...)
        """

        super().__init__(image_dimensions=image_dimensions)
        self.num_classes = num_classes

        conv3d_transpose_paddings = calculate_required_paddings(
            *self.image_dimensions.get_dhw(), num_levels=4
        )
        self.layers = UNet3dParts.build_layers(
            self.image_dimensions.channels, num_classes, conv3d_transpose_paddings
        )

    def forward(self, x):
        encoder_outputs = self.__encode(x)

        not_bottleneck_output = self.layers["not_bottleneck"](encoder_outputs[-1])
        not_bottleneck_upsampled = self.layers["upsampling_level_4"](
            not_bottleneck_output
        )

        decoder_output = self.__decode(encoder_outputs, not_bottleneck_upsampled)
        decoder_output_adjusted_channels = self.layers["output_conv"](decoder_output)
        return self.layers["output_activation"](decoder_output_adjusted_channels)

    def __decode(self, encoder_outputs, not_bottleneck_upsampled):
        prev_level_upsampled = not_bottleneck_upsampled
        for level in range(3, 0, -1):
            decoder_level_output = self.__decode_single(
                encoder_outputs[level - 1], prev_level_upsampled, level
            )
            prev_level_upsampled = decoder_level_output
        # noinspection PyUnboundLocalVariable
        return decoder_level_output

    def __encode(self, x):
        input = x
        outputs = []
        for level in range(1, 4):
            encoder_level_output = self.layers[f"encoder_level_{level}"](input)
            outputs.append(encoder_level_output)
            encoder_level_pooled = self.layers["max_pool"](encoder_level_output)
            input = encoder_level_pooled
        # noinspection PyUnboundLocalVariable
        outputs.append(encoder_level_pooled)
        return outputs

    def __decode_single(
        self, encoder_level_n_output, decoder_level_n_minus_one_upsampled, level
    ):
        decoder_level_n_input = torch.concat(
            (encoder_level_n_output, decoder_level_n_minus_one_upsampled), axis=1
        )
        decoder_level_n_output = self.layers[f"decoder_level_{level}"](
            decoder_level_n_input
        )
        if level > 1:
            decoder_level_n_upsampled = self.layers[f"upsampling_level_{level}"](
                decoder_level_n_output
            )
            return decoder_level_n_upsampled
        return decoder_level_n_output
