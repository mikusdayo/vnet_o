mkdir data && cd data
kaggle datasets download -d mateuszbuda/lgg-mri-segmentation
unzip -q lgg-mri-segmentation.zip
rm lgg-mri-segmentation.zip
rm -r lgg-mri-segmentation