set -e 

batch_size=64
echo "Parameters: Batch size is set to $batch_size"

echo "Begin Evaluation\n"

for mode in "dev" "test" ; do
    echo "Evaluating on msmarco ${mode} set\n"
    for dataset in "doc" "passage"; do
        for m in 96 64 48 32 24 16; do
            echo "Running inference for msmarco-${dataset} ${mode} dataset"
            echo "Parameters: Number of subvectors for each ${dataset} is set to ${m}"
            index_path="./data/${dataset}/download_jpq_index/OPQ${m},IVF1,PQ${m}x8.index"
            query_encoder_path="./data/${dataset}/download_query_encoder/m${m}"

            preprocess_dir="./data/${dataset}/preprocess"
            output_path=./data/$dataset/run_retrieve/${mode}/run.${mode}.m${m}.rank
            if [ $m -ge 56 ]
            then
                echo "Use cpu search"
                python ./run_retrieval.py \
                --preprocess_dir $preprocess_dir \
                --index_path $index_path \
                --mode $mode \
                --query_encoder_dir $query_encoder_path \
                --output_path $output_path \
                --batch_size $batch_size 
            else 
                echo "Use gpu searchp"
                python ./run_retrieval.py \
                --preprocess_dir $preprocess_dir \
                --index_path $index_path \
                --mode $mode \
                --query_encoder_dir $query_encoder_path \
                --output_path $output_path \
                --batch_size $batch_size \
                --gpu_search
            fi
            
            
            # evaluation
            label_path=./data/${dataset}/preprocess/$mode-qrel.tsv 
            if [ $mode = "dev" ]
            then 
                if [ $dataset = "passage" ] 
                then
                    python ./msmarco_eval.py $label_path $output_path
                else
                    python ./msmarco_eval.py $label_path $output_path 100
                fi
            else
                ./data/trec_eval-9.0.7/trec_eval -c -mrecall.100 -mndcg_cut.10 $label_path $output_path
            fi
            
            echo "Convert qids and pids to official ids"
            official_id_rank_path=./data/$dataset/run_retrieve/${mode}/official.run.${mode}.m${m}.rank
            python ./cvt_back.py \
                --input_path $output_path \
                --preprocess_dir $preprocess_dir \
                --mode $mode \
                --output_path $official_id_rank_path \
                --dataset $dataset 

            echo "Use official qrels files to compute metrics"
            if [ $mode = "dev" ]
            then 
            # Evaluate MSMARCO Dev set
                if [ $dataset = "passage" ] 
                then
                    python ./msmarco_eval.py ./data/passage/dataset/qrels.dev.small.tsv $official_id_rank_path
                else
                    python ./msmarco_eval.py ./data/doc/dataset/msmarco-docdev-qrels.tsv $official_id_rank_path doc
                fi
            else
            # Evaluate TREC Test
                if [ $dataset = "passage" ] 
                then
                    ./data/trec_eval-9.0.7/trec_eval -c -mndcg_cut.10 -mrecall.100 ./data/passage/dataset/2019qrels-pass.txt $official_id_rank_path
                else
                    ./data/trec_eval-9.0.7/trec_eval -c -mndcg_cut.10 -mrecall.100 ./data/doc/dataset/2019qrels-docs.txt $official_id_rank_path
                fi
            fi
            echo "End experiment for m=$m"
            echo "***************************\n"
        done
    done
done
