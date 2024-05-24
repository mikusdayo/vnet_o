from setuptools import setup, find_packages


def parse_requirements():
    with open("requirements.txt", "r") as f:
        return f.read().split("\n")


if __name__ == "__main__":
    setup(
        name="volseg",
        version="1.0.0",
        install_requires=parse_requirements(),
        packages=find_packages(),
        url="https://github.com/bwieciech/volumetric_segmentation",
        license="MIT",
        author="Bartosz Wieciech",
        author_email="bartek.wieciech@gmail.com",
        description="PyTorch implementation of VNet and 3D UNet for volumetric segmentation",
        python_requires=">=3.7.1",
    )
