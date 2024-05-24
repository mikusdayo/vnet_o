from collections import defaultdict


def calculate_required_paddings(im_depth, im_height, im_width, num_levels):
    """
    Calculates paddings required in ConvTranspose3d layers. These paddings are needed when stride 2 max pooling is
    applied to a dimension of odd size. The resulting layer is of size floor(dims / 2). Upsampling this layer produces
    an output of even size, hence there is a possibility of dimension mismatch.
    :return: dictionary of form level_number -> (depth_padding_px, height_padding_px, width_padding_px)
    """
    res = defaultdict(lambda: (0, 0, 0))

    depth, height, width = im_depth, im_height, im_width
    for layer in range(
        2, num_levels + 1
    ):  # potential upscaling mismatch occurs in levels 2+
        up_depth, up_height, up_width = (
            2 * (depth // 2),
            2 * (height // 2),
            2 * (width // 2),
        )
        res[layer] = (depth - up_depth, height - up_height, width - up_width)
        depth, height, width = depth // 2, height // 2, width // 2

    return res
