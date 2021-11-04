set -e 

root_dir="./data"

# download for passage dataset

ivf_passage_dir="${root_dir}/passage/download_jpq_index"
mkdir -p $ivf_passage_dir
cd $ivf_passage_dir 

wget https://www.dropbox.com/s/hcgry8oljoif4wo/OPQ96%2CIVF1%2CPQ96x8.index?dl=0 -O OPQ96,IVF1,PQ96x8.index

wget https://www.dropbox.com/s/yz1wzyslv0me7cp/OPQ64%2CIVF1%2CPQ64x8.index?dl=0 -O OPQ64,IVF1,PQ64x8.index

wget https://www.dropbox.com/s/49wzuvs6z506z5u/OPQ48%2CIVF1%2CPQ48x8.index?dl=0 -O OPQ48,IVF1,PQ48x8.index

wget https://www.dropbox.com/s/ynrh8x97ndd083z/OPQ32%2CIVF1%2CPQ32x8.index?dl=0 -O OPQ32,IVF1,PQ32x8.index

wget https://www.dropbox.com/s/78054st0b5tfqis/OPQ24%2CIVF1%2CPQ24x8.index?dl=0 -O OPQ24,IVF1,PQ24x8.index

wget https://www.dropbox.com/s/ge41slv35s82n0s/OPQ16%2CIVF1%2CPQ16x8.index?dl=0 -O OPQ16,IVF1,PQ16x8.index

echo "Finish downloading all pq indexes for msmarco-passage dataset"

# download for document dataset

cd ../../
ivf_doc_dir="doc/download_jpq_index"
mkdir -p $ivf_doc_dir
cd $ivf_doc_dir

wget https://www.dropbox.com/s/6yk3hy4oq892u1v/OPQ96%2CIVF1%2CPQ96x8.index?dl=0 -O OPQ96,IVF1,PQ96x8.index

wget https://www.dropbox.com/s/x803xiqtvdap587/OPQ64%2CIVF1%2CPQ64x8.index?dl=0 -O OPQ64,IVF1,PQ64x8.index

wget https://www.dropbox.com/s/ljit0lhm245osc2/OPQ48%2CIVF1%2CPQ48x8.index?dl=0 -O OPQ48,IVF1,PQ48x8.index

wget https://www.dropbox.com/s/kyalxe3kidtajgq/OPQ32%2CIVF1%2CPQ32x8.index?dl=0 -O OPQ32,IVF1,PQ32x8.index

wget https://www.dropbox.com/s/xhfho87miqi14d5/OPQ24%2CIVF1%2CPQ24x8.index?dl=0 -O OPQ24,IVF1,PQ24x8.index

wget https://www.dropbox.com/s/83kfs6tg2u51m3i/OPQ16%2CIVF1%2CPQ16x8.index?dl=0 -O OPQ16,IVF1,PQ16x8.index

echo "Finish downloading all pq indexes for msmarco-doc dataset"




