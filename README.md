# JPQ

Repo for our CIKM'21 Full paper, [Jointly Optimizing Query Encoder and Product Quantization to Improve Retrieval Performance](https://arxiv.org/abs/2108.00644). JPQ greatly improves the efficiency of Dense Retrieval. It is able to compress the index size by 30x with negligible performance loss. It also provides 10x speedup on CPU and 2x speedup on GPU in query latency. 

Here is the effectiveness - index size (log-scale) tradeoff on MSMARCO Passage Ranking. **In contrast with trading index size for ranking performance, JPQ achieves high ranking effectiveness with a tiny index.** 
<p align="center">
<img src="./figures/all_psg_compr.png" width="50%" height="50%">
</p>

Results at different trade-off settings are shown below.

MS MARCO Passage Ranking   |  MS MARCO Document Ranking
:-------------------------:|:-------------------------:
<img src="./figures/psg_compr.png" width="70%">  | <img src="./figures/doc_compr.png" width="70%"> 

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

## Data

You can download trained models and indexes from our [dropbox link](https://www.dropbox.com/sh/miczl0zlj8vy47v/AAAwTNus2g6sABLB4dVH3Adba?dl=0). After open this link in your browser, you can see two folder, `doc` and `passage`. They correspond to MSMARCO passage ranking and document ranking. There are also two folders in either of them, `trained_models` and `indexes`. `trained_models` are the trained query encoders, and `indexes` are trained PQ indexes. Note, the `pid` in the index is actually the row number of a passage in the `collection.tsv` file instead of the official pid provided by MS MARCO. Different query encoders and indexes correspond to different compression ratios. For example, the query encoder named `m32.tar.gz` or the index named `OPQ32,IVF1,PQ32x8.index` means 32 bytes per doc, i.e., `768*4/32=96x` compression ratio.

## Retrieval
Instructions will be available soon.

## Training
Instructions will be available soon.
