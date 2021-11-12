set -e 

data_dir="./data/passage/download_dual_encoders"
mkdir -p $data_dir
cd $data_dir

wget https://www.dropbox.com/s/cdbpuejt0t63csg/m96.tar.gz?dl=0 -O m96.tar.gz
tar -xzvf m96.tar.gz

wget https://www.dropbox.com/s/5tbl79jo10v6jjs/m64.tar.gz?dl=0 -O m64.tar.gz
tar -xzvf m64.tar.gz

wget https://www.dropbox.com/s/ajho1scx2ld55ik/m48.tar.gz?dl=0 -O m48.tar.gz
tar -xzvf m48.tar.gz

wget https://www.dropbox.com/s/9hgziqi5sx5qxrb/m32.tar.gz?dl=0 -O m32.tar.gz
tar -xzvf m32.tar.gz

wget https://www.dropbox.com/s/td5lue4in3d7l9z/m24.tar.gz?dl=0 -O m24.tar.gz
tar -xzvf m24.tar.gz

wget https://www.dropbox.com/s/lzshaejrt1d5bb3/m16.tar.gz?dl=0 -O m16.tar.gz
tar -xzvf m16.tar.gz

echo "Finish downloading all dual-encoders trained on msmarco-passage dataset"




