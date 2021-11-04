set -e 

root_dir="./data"

# download for passage dataset

encoder_passage_dir="${root_dir}/passage/download_query_encoder"
mkdir -p $encoder_passage_dir
cd $encoder_passage_dir 

wget https://www.dropbox.com/s/xo1hlbgg7yi7s6a/m96.tar.gz?dl=0 -O m96.tar.gz
tar -xzvf m96.tar.gz

wget https://www.dropbox.com/s/ojg8yhh8og8rq19/m64.tar.gz?dl=0 -O m64.tar.gz
tar -xzvf m64.tar.gz

wget https://www.dropbox.com/s/sftzgdpi84zjizm/m48.tar.gz?dl=0 -O m48.tar.gz
tar -xzvf m48.tar.gz

wget https://www.dropbox.com/s/u90ca9dmtme7s30/m32.tar.gz?dl=0 -O m32.tar.gz
tar -xzvf m32.tar.gz

wget https://www.dropbox.com/s/1zk3thnoov7f8gj/m24.tar.gz?dl=0 -O m24.tar.gz
tar -xzvf m24.tar.gz

wget https://www.dropbox.com/s/to1ejx6lypwws4z/m16.tar.gz?dl=0 -O m16.tar.gz
tar -xzvf m16.tar.gz

echo "Finish downloading all pq indexes for msmarco-passage dataset"

# download for document dataset

cd ../../
encoder_doc_dir="doc/download_query_encoder"
mkdir -p $encoder_doc_dir
cd $encoder_doc_dir

wget https://www.dropbox.com/s/lmcrul04imp1t1e/m96.tar.gz?dl=0 -O m96.tar.gz
tar -xzvf m96.tar.gz

wget https://www.dropbox.com/s/tqtln72uhk8bp2n/m64.tar.gz?dl=0 -O m64.tar.gz
tar -xzvf m64.tar.gz

wget https://www.dropbox.com/s/gl12r8mqak3rpr3/m48.tar.gz?dl=0 -O m48.tar.gz
tar -xzvf m48.tar.gz

wget https://www.dropbox.com/s/8nq1ytx6cp7qija/m32.tar.gz?dl=0 -O m32.tar.gz
tar -xzvf m32.tar.gz

wget https://www.dropbox.com/s/zf66buq6rplwraq/m24.tar.gz?dl=0 -O m24.tar.gz
tar -xzvf m24.tar.gz

wget https://www.dropbox.com/s/kt0sax5w05jz69x/m16.tar.gz?dl=0 -O m16.tar.gz
tar -xzvf m16.tar.gz

echo "Finish downloading all pq indexes for msmarco-doc dataset"




