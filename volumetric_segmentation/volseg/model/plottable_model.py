import torch
from torchviz import make_dot

from volseg.utils.image_dimension_wrapper import ImageDimensionsWrapper
from volseg.utils.io_utils import print_info_message


class PlottableModel(torch.nn.Module):
    def __init__(self, image_dimensions):
        super().__init__()
        self.image_dimensions = ImageDimensionsWrapper(dims=image_dimensions)

    def visualize(self):
        print_info_message(
            "In case of \"Not a directory: PosixPath('dot')\" error, install graphviz manually, "
            "e.g. through apt."
        )

        input = torch.zeros(
            1, *self.image_dimensions.get(), dtype=torch.float, requires_grad=False
        )
        output = self(input)
        return make_dot(
            output.mean(),
            params=dict(self.named_parameters()),
            show_attrs=True,
            show_saved=True,
        )
