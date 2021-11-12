set -e 

batch_size=64
echo "Parameters: Batch size is set to $batch_size"

echo "Begin Evaluation\n"

for year in "2020" "2019"  ; do
    echo "Evaluating on msmarco TREC${year} set\n"
    for dataset in "doc" "passage"; do
        for m in 96 64 48 32 24 16; do
            echo "Running inference for msmarco-${dataset} TREC${year} dataset"
            echo "Parameters: Number of subvectors for each ${dataset} is set to ${m}"
            index_path="./data/${dataset}/download_jpq_index/OPQ${m},IVF1,PQ${m}x8.index"
            query_encoder_path="./data/${dataset}/download_query_encoder/m${m}"

            trec_data_dir="./data/${dataset}/trec20-test"
            query_file_path="${trec_data_dir}/msmarco-test${year}-queries.tsv "
            pid2offset_path="./data/${dataset}/preprocess/pid2offset.pickle"
            output_path=./data/$dataset/tokenize_retrieve/official.run.trec${year}.m${m}.rank

            if [ $m -ge 56 ]
            then
                echo "Use cpu search"
                search_option=""
            else 
                echo "Use gpu search"
                search_option="--gpu_search"
            fi

            python -m jpq.tokenize_retrieve \
                --query_file_path $query_file_path \
                --index_path $index_path \
                --query_encoder_dir $query_encoder_path \
                --output_path $output_path \
                --output_format "trec" \
                --pid2offset_path $pid2offset_path \
                --dataset $dataset \
                --batch_size $batch_size \
                $search_option

            echo "Use official qrels files to compute metrics"
            
            # Evaluate TREC Test
            if [ $dataset = "passage" ] 
            then
                ./data/trec_eval-9.0.7/trec_eval -c -mndcg_cut.10 -mrecall.100 $trec_data_dir/${year}qrels-pass.txt $output_path
            else
                ./data/trec_eval-9.0.7/trec_eval -c -mndcg_cut.10 -mrecall.100 $trec_data_dir/${year}qrels-docs.txt $output_path
            fi
            
            echo "End experiment for m=$m"
            echo "***************************\n"
        done
    done
done
