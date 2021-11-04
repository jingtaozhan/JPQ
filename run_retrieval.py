# coding=utf-8
import os
from urllib import parse
import torch
import faiss
import logging
import argparse
import numpy as np
from tqdm import tqdm
from transformers import RobertaConfig
from model import RobertaDot
from timeit import default_timer as timer
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.sampler import SequentialSampler

from dataset import TextTokenIdsCache, SequenceDataset, get_collate_function

logger = logging.Logger(__name__, level=logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s-%(name)s-%(levelname)s- %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

def load_index(index_path, use_cuda, faiss_gpu_index):
    index = faiss.read_index(index_path)
    if use_cuda:
        res = faiss.StandardGpuResources()
        res.setTempMemory(1024*1024*1024)
        co = faiss.GpuClonerOptions()
        if isinstance(index, faiss.IndexPreTransform): 
            subvec_num = faiss.downcast_index(index.index).pq.M
        else:
            subvec_num = index.pq.M
        if int(subvec_num) >= 56:
            co.useFloat16 = True
        else:
            co.useFloat16 = False
        logger.info(f"subvec_num: {subvec_num}; useFloat16: {co.useFloat16}")
        if co.useFloat16:
            logger.warning("If the number of subvectors >= 56 and gpu search is turned on, Faiss uses float16 and therefore there is a little performance loss. You can use cpu search to obtain the best ranking effectiveness")
        index = faiss.index_cpu_to_gpu(res, faiss_gpu_index, index, co)
    return index


def query_inference(model, index, args):
    query_dataset = SequenceDataset(
        ids_cache=TextTokenIdsCache(data_dir=args.preprocess_dir, prefix=f"{args.mode}-query"),
        max_seq_length=args.max_query_length
    )
    
    model = model.to(args.device)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)
    
    dataloader = DataLoader(
        query_dataset,
        sampler=SequentialSampler(query_dataset),
        batch_size=args.batch_size,
        collate_fn=get_collate_function(args.max_query_length),
        drop_last=False,
    )
    batch_size = dataloader.batch_size
    num_examples = len(dataloader.dataset)
    logger.info("  Num examples = %d", num_examples)
    logger.info("  Batch size = %d", batch_size)

    model.eval()

    all_search_results_pids, all_search_results_scores = [], []
    for inputs, ids in tqdm(dataloader):
        for k, v in inputs.items():
            if isinstance(v, torch.Tensor):
                inputs[k] = v.to(args.device)
        with torch.no_grad():
            query_embeds = model(is_query=True, **inputs).detach().cpu().numpy()
            batch_results_scores, batch_results_pids = index.search(query_embeds, args.topk)
            all_search_results_pids.extend(batch_results_pids.tolist())
            all_search_results_scores.extend(batch_results_scores.tolist())
    return all_search_results_scores, all_search_results_pids


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--preprocess_dir", type=str, required=True)
    parser.add_argument("--mode", type=str, choices=["dev", "test"], required=True)
    parser.add_argument("--index_path", type=str, required=True)
    parser.add_argument("--query_encoder_dir", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--max_query_length", type=int, default=32)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--topk", type=int, default=100)
    parser.add_argument("--gpu_search", action="store_true")
    args = parser.parse_args()

    args.device = torch.device("cuda:0")
    args.n_gpu = 1
    
    config_class, model_class = RobertaConfig, RobertaDot
    
    config = config_class.from_pretrained(args.query_encoder_dir)
    model = model_class.from_pretrained(args.query_encoder_dir, config=config,)
    index = load_index(args.index_path, use_cuda=args.gpu_search, faiss_gpu_index=0)
    if not args.gpu_search:
        faiss.omp_set_num_threads(32)
    all_search_results_scores, all_search_results_pids = query_inference(model, index, args)

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    
    with open(args.output_path, 'w') as outputfile:
        for qid, (scores, pids) in enumerate(zip(all_search_results_scores, all_search_results_pids)):
            for idx, (score, pid) in enumerate(zip(scores, pids)):
                rank = idx+1
                if args.mode == "dev":
                    outputfile.write(f"{qid}\t{pid}\t{rank}\n")
                else:
                    assert args.mode == "test" # TREC Test
                    index_name = os.path.basename(args.index_path)
                    outputfile.write(f"{qid} Q0 {pid} {rank} {score} JPQ-{index_name}\n")


if __name__ == "__main__":
    main()
