# JPQ

* 🔥**News 2021-10: The extension of JPQ, [Learning Discrete Representations via Constrained Clustering for Effective and Efficient Dense Retrieval](https://arxiv.org/abs/2110.05789) \[[code](https://github.com/jingtaozhan/RepCONC)\], was accepted by WSDM'22. It presents RepCONC and achieves state-of-the-art first-stage retrieval effectiveness-efficiency tradeoff. It utilizes constrained clustering to train discrete codes and then incorporates JPQ in the second-stage training.**

Repo for our CIKM'21 Full paper, [Jointly Optimizing Query Encoder and Product Quantization to Improve Retrieval Performance](https://arxiv.org/abs/2108.00644) ([poster](https://drive.google.com/file/d/1UxTg4sm0ffnZmqVutJzolE5hDRKGzgOQ/view?usp=sharing), [presentation record](https://www.youtube.com/watch?v=kZmEtLtn1PU&t=3s)). JPQ greatly improves the efficiency of Dense Retrieval. It is able to compress the index size by 30x with negligible performance loss. It also provides 10x speedup on CPU and 2x speedup on GPU in query latency. 

Here is the effectiveness - index size (log-scale) tradeoff on MSMARCO Passage Ranking. **In contrast with trading index size for ranking performance, JPQ achieves high ranking effectiveness with a tiny index.** 
<p align="center">
<img src="./figures/all_psg_compr.png" width="50%" height="50%">
</p>

Results at different trade-off settings are shown below.

MS MARCO Passage Ranking   |  MS MARCO Document Ranking
:-------------------------:|:-------------------------:
<img src="./figures/psg_compr.png" width="60%">  | <img src="./figures/doc_compr.png" width="60%"> 

JPQ is still very effective even if the compression ratio is over 100x and outperforms baselines at different compression ratio settings. 
For more details, please refer to our paper. If you find this repo useful, please do not save your star and cite our work:

```
@article{zhan2021jointly,
  title={Jointly Optimizing Query Encoder and Product Quantization to Improve Retrieval Performance},
  author={Zhan, Jingtao and Mao, Jiaxin and Liu, Yiqun and Guo, Jiafeng and Zhang, Min and Ma, Shaoping},
  journal={arXiv preprint arXiv:2108.00644},
  year={2021}
}
```

## Models and Indexes

You can download trained models and indexes from our [dropbox link](https://www.dropbox.com/sh/miczl0zlj8vy47v/AAAwTNus2g6sABLB4dVH3Adba?dl=0). After open this link in your browser, you can see two folder, `doc` and `passage`. They correspond to MSMARCO passage ranking and document ranking. There are also two folders in either of them, `trained_models` and `indexes`. `trained_models` are the trained query encoders, and `indexes` are trained PQ indexes. Note, the `pid` in the index is actually the row number of a passage in the `collection.tsv` file instead of the official pid provided by MS MARCO. Different query encoders and indexes correspond to different compression ratios. For example, the query encoder named `m32.tar.gz` or the index named `OPQ32,IVF1,PQ32x8.index` means 32 bytes per doc, i.e., `768*4/32=96x` compression ratio.

## Requirements

To install requirements, run the following commands:

```setup
git clone git@github.com:jingtaozhan/JPQ.git
cd JPQ
python setup.py install
```

## Preprocess
Here are the commands to for preprocessing/tokenization. 

If you do not have MS MARCO dataset, run the following command:
```
bash download_data.sh
```
Preprocessing (tokenizing) only requires a simple command:
```
python preprocess.py --data_type 0; python preprocess.py --data_type 1
```
It will create two directories, i.e., `./data/passage/preprocess` and `./data/doc/preprocess`. We map the original qid/pid to new ids, the row numbers in the file. The mapping is saved to `pid2offset.pickle` and `qid2offset.pickle`, and new qrel files (`train/dev/test-qrel.tsv`) are generated. The passages and queries are tokenized and saved in the numpy memmap file. 

Note: JPQ, as long as our [SIGIR'21 models](https://github.com/jingtaozhan/DRhard), utilizes Transformers 2.x version to tokenize text. However, when Transformers library updates to 3.x or 4.x versions, the RobertaTokenizer behaves differently. 
To support REPRODUCIBILITY, we copy the RobertaTokenizer source codes from 2.x version to [star_tokenizer.py](https://github.com/jingtaozhan/JPQ/blob/main/star_tokenizer.py). During preprocessing, we use `from star_tokenizer import RobertaTokenizer` instead of `from transformers import RobertaTokenizer`. It is also **necessary** for you to do this if you use our JPQ model on other datasets. 

## Retrieval

You can download the query encoders and indexes from our [dropbox link](https://www.dropbox.com/sh/miczl0zlj8vy47v/AAAwTNus2g6sABLB4dVH3Adba?dl=0) and run the following command to efficiently retrieve documents:
```
python ./run_retrieval.py \
    --preprocess_dir ./data/doc/preprocess \
    --mode dev \
    --index_path PATH/TO/OPQ96,IVF1,PQ96x8.index \
    --query_encoder_dir PATH/TO/m96/ \
    --output_path ./data/doc/m96.dev.tsv \
    --batch_size 32 \
    --topk 100
```
It also has an option `--gpu_search` for fast GPU search.

Run the following command to evaluate the ranking results on MSMARCO document dataset.
```bash
python ./msmarco_eval.py ./data/doc/preprocess/dev-qrel.tsv ./data/doc/m96.dev.tsv 100
```
You will get
```bash
Eval Started
#####################
MRR @100: 0.4008114949611788
QueriesRanked: 5193
#####################
```

## Training

JPQ is initialized by [STAR](https://github.com/jingtaozhan/DRhard). STAR trained on passage ranking is available [here](https://drive.google.com/drive/folders/1bJw8P15cFiV239mTgFQxVilXMWqzqXUU?usp=sharing). STAR trained on document ranking is available [here](https://drive.google.com/drive/folders/18GrqZxeiYFxeMfSs97UxkVHwIhZPVXTc?usp=sharing). 

First, use STAR to encode the corpus and run OPQ to initialize the index. For example, on document ranking task, please run:
```bash
python ./run_init.py \
  --preprocess_dir ./data/doc/preprocess/ \
  --model_dir ./data/doc/star \
  --max_doc_length 512 \
  --output_dir ./data/doc/init \
  --subvector_num 96
```
On passage ranking task, you can set the `max_doc_length` to 256 for faster inference. 

Now you can train the query encoder and PQ index. For example, on document ranking task, the command is 
```bash
python run_train.py \
    --preprocess_dir ./data/doc/preprocess \
    --model_save_dir ./data/doc/train/m96/models \
    --log_dir ./data/doc/train/m96/log \
    --init_index_path ./data/doc/init/OPQ96,IVF1,PQ96x8.index \
    --init_model_path ./data/doc/star \
    --lambda_cut 10 \
    --centroid_lr 1e-4 \
    --train_batch_size 32
```
`--gpu_search` is optional for fast gpu search during training. `lambda_cut` should be set to 200 for passage ranking task. `centroid_lr` is different for different compression ratios. 
Let M be the number of subvectors. `centroid_lr` equals to 5e-6 for  `M = 16/24`, 2e−5 for `M = 32`, and 1e−4 for `M = 48/64/96`. The number of training epochs is set to 6. In fact, the performance is already quite satisfying after 1 or 2 epochs. Each epoch costs less than 2 hours on our machine. 




