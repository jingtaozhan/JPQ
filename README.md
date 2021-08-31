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
Trained models and indexes will be available soon.

## Retrieval
Instructions will be available soon.

## Training
Instructions will be available soon.
