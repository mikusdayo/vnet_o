class ImageDimensionsWrapper:
    def __init__(self, dims):
        if isinstance(dims, self.__class__):
            self.channels, self.depth, self.height, self.width = (
                dims.channels,
                dims.depth,
                dims.height,
                dims.width,
            )
        else:
            self.channels, self.depth, self.height, self.width = dims

    def get(self):
        return self.channels, self.depth, self.height, self.width

    def get_dhw(self):
        return self.depth, self.height, self.width
