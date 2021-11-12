set -e

dataset=$1 # "nfcorpus" "scifact" "arguana" "scidocs" "fiqa" "trec-covid" ...
M=${2:-96} # number of sub-vectors
split=${3:-"test"} # test/dev/train
encode_batch_size=${4:-64}

echo dataset: $dataset
echo M: $M
echo split: $split

beir_data_root="./data/beir"
output_dir="./data/passage/beir_output/m${M}/${dataset}"
model_root="./data/passage/download_dual_encoders"
query_encoder="${model_root}/m${M}/m${M}.marcopass.query.encoder"
doc_encoder="${model_root}/m${M}/m${M}.marcopass.doc.encoder"

python -m jpq.eval_beir \
    --dataset $dataset \
    --beir_data_root $beir_data_root \
    --split $split \
    --encode_batch_size $encode_batch_size \
    --query_encoder $query_encoder \
    --doc_encoder $doc_encoder \
    --output_index_path "${output_dir}/index" \
    --output_ranking_path "${output_dir}/${split}-ranking.pickle"


