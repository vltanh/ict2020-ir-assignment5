mkdir -p data
cd data
wget https://www.robots.ox.ac.uk/~vgg/data/oxbuildings/oxbuild_images.tgz
wget https://www.robots.ox.ac.uk/~vgg/data/oxbuildings/gt_files_170407.tgz
untar xvf oxbuild_images.tgz
untar xvf gt_files_170407.tgz