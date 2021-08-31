# coding=utf-8
'''
Encoding part is modified base on 
    https://github.com/jingtaozhan/DRhard/blob/main/star/inference.py (SIGIR'21)
'''
import os
import torch
import faiss
import argparse
import subprocess
import logging
import numpy as np
from tqdm import tqdm
from transformers import RobertaConfig
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.sampler import SequentialSampler

from model import RobertaDot
from dataset import (
    TextTokenIdsCache, SequenceDataset,
    get_collate_function
)
logger = logging.Logger(__name__, level=logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s-%(name)s-%(levelname)s- %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

def prediction(model, data_collator, args, test_dataset, embedding_memmap, is_query):
    os.makedirs(args.output_dir, exist_ok=True)
    test_dataloader = DataLoader(
        test_dataset,
        sampler=SequentialSampler(test_dataset),
        batch_size=args.eval_batch_size*args.n_gpu,
        collate_fn=data_collator,
        drop_last=False,
    )
    # multi-gpu eval
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    batch_size = test_dataloader.batch_size
    num_examples = len(test_dataloader.dataset)
    logger.info("***** Running *****")
    logger.info("  Num examples = %d", num_examples)
    logger.info("  Batch size = %d", batch_size)

    model.eval()
    write_index = 0
    for step, (inputs, ids) in enumerate(tqdm(test_dataloader)):
        for k, v in inputs.items():
            if isinstance(v, torch.Tensor):
                inputs[k] = v.to(args.device)
        with torch.no_grad():
            logits = model(is_query=is_query, **inputs).detach().cpu().numpy()
        write_size = len(logits)
        assert write_size == len(ids)
        embedding_memmap[write_index:write_index+write_size] = logits
        write_index += write_size
    assert write_index == len(embedding_memmap)


def doc_inference(model, args, embedding_size):
    doc_collator = get_collate_function(args.max_doc_length)
    ids_cache = TextTokenIdsCache(data_dir=args.preprocess_dir, prefix="passages")
    doc_dataset = SequenceDataset(
        ids_cache=ids_cache,
        max_seq_length=args.max_doc_length
    )
    assert not os.path.exists(args.doc_embed_path)
    doc_memmap = np.memmap(args.doc_embed_path, 
        dtype=np.float32, mode="w+", shape=(len(doc_dataset), embedding_size))
    try:
        prediction(model, doc_collator, args,
            doc_dataset, doc_memmap, is_query=False
        )
    except:
        subprocess.check_call(["rm", args.doc_embed_path])
        raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--preprocess_dir", type=str, required=True)
    parser.add_argument("--model_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--subvector_num", type=int, required=True)
    parser.add_argument("--max_doc_length", type=int, default=512)
    parser.add_argument("--eval_batch_size", type=int, default=128)
    args = parser.parse_args()

    args.device = torch.device("cuda")
    args.n_gpu = torch.cuda.device_count()
    logger.info(args)

    os.makedirs(args.output_dir, exist_ok=True)
    args.doc_embed_path = os.path.join(args.output_dir, "doc_embed.memmap")
    embed_size = 768

    if not os.path.exists(args.doc_embed_path):
        logger.info("Encoding passages to dense vectors ...")
        config = RobertaConfig.from_pretrained(args.model_dir, gradient_checkpointing=False)
        model = RobertaDot.from_pretrained(args.model_dir, config=config)
        model = model.to(args.device)
        doc_inference(model, args, embed_size)
        model = None
        torch.cuda.empty_cache()
    else:
        logger.info(f"{args.doc_embed_path} exists, skip encoding procedure")

    doc_embeddings = np.memmap(args.doc_embed_path, 
        dtype=np.float32, mode="r")
    doc_embeddings = doc_embeddings.reshape(-1, embed_size)

    save_index_path = os.path.join(args.output_dir, f"OPQ{args.subvector_num},IVF1,PQ{args.subvector_num}x8.index")
    res = faiss.StandardGpuResources()
    res.setTempMemory(1024*1024*512)
    co = faiss.GpuClonerOptions()
    co.useFloat16 = args.subvector_num >= 56

    faiss.omp_set_num_threads(32)
    dim = embed_size
    index = faiss.index_factory(dim, 
        f"OPQ{args.subvector_num},IVF1,PQ{args.subvector_num}x8", faiss.METRIC_INNER_PRODUCT)
    index.verbose = True
    index = faiss.index_cpu_to_gpu(res, 0, index, co)    
    index.train(doc_embeddings)
    index.add(doc_embeddings)
    index = faiss.index_gpu_to_cpu(index)
    faiss.write_index(index, save_index_path)


if __name__ == "__main__":
    main()
