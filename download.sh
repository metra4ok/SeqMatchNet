# download nordland-clean dataset
wget -cO - https://cloudstor.aarnet.edu.au/plus/s/PK98pDvLAesL1aL/download > nordland-clean.zip
mkdir -p ./data/
unzip nordland-clean.zip -d ./data/
rm nordland-clean.zip

# download oxford descriptors
wget -cO - https://cloudstor.aarnet.edu.au/plus/s/T0M1Ry4HXOAkkGz/download > oxford_2014-12-16-18-44-24_stereo_left.npy
wget -cO - https://cloudstor.aarnet.edu.au/plus/s/vr21RnhMmOkW8S9/download > oxford_2015-03-17-11-08-44_stereo_left.npy
mv oxford* ./data/descData/netvlad-pytorch/