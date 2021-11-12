
import os
import torch
import random
import time
import faiss
import logging
import argparse
import numpy as np
from tqdm import tqdm, trange
from collections import defaultdict
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, RandomSampler
from transformers import (AdamW, get_linear_schedule_with_warmup,
    RobertaConfig)

from jpq.dataset import TextTokenIdsCache, SequenceDataset, pack_tensor_2D
from jpq.model import RobertaDot

logger = logging.Logger(__name__, level=logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s-%(name)s-%(levelname)s- %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def save_model(model, output_dir, save_name, args, optimizer=None):
    save_dir = os.path.join(output_dir, save_name)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    model_to_save = model.module if hasattr(model, 'module') else model  
    model_to_save.save_pretrained(save_dir)
    torch.save(args, os.path.join(save_dir, 'training_args.bin'))
    if optimizer is not None:
        torch.save(optimizer.state_dict(), os.path.join(save_dir, "optimizer.bin"))



class TrainQueryDataset(SequenceDataset):
    def __init__(self, queryids_cache, 
            rel_file, max_query_length):
        SequenceDataset.__init__(self, queryids_cache, max_query_length)
        self.reldict = defaultdict(list)
        for line in tqdm(open(rel_file), desc=os.path.split(rel_file)[1]):
            qid, _, pid, _ = line.split()
            qid, pid = int(qid), int(pid)
            self.reldict[qid].append((pid))

    def __getitem__(self, item):
        ret_val = super().__getitem__(item)
        ret_val['rel_ids'] = self.reldict[item]
        return ret_val


def get_collate_function(max_seq_length):
    cnt = 0
    def collate_function(batch):
        nonlocal cnt
        length = None
        if cnt < 10:
            length = max_seq_length
            cnt += 1

        input_ids = [x["input_ids"] for x in batch]
        attention_mask = [x["attention_mask"] for x in batch]
        data = {
            "input_ids": pack_tensor_2D(input_ids, default=1, 
                dtype=torch.int64, length=length),
            "attention_mask": pack_tensor_2D(attention_mask, default=0, 
                dtype=torch.int64, length=length),
        }
        qids = [x['id'] for x in batch]
        all_rel_pids = [x["rel_ids"] for x in batch]
        return data, all_rel_pids
    return collate_function  
    

def get_doc_embeds(psg_ids, pq_codes, centroids):
    M = centroids.shape[0]
    first_indices = torch.arange(M).to(psg_ids.device)
    first_indices = first_indices.expand(len(psg_ids), M).reshape(-1)
    second_indices = pq_codes[psg_ids].reshape(-1)
    embeddings = centroids[first_indices, second_indices].reshape(len(psg_ids), -1)
    return embeddings


def compute_loss(query_embeddings, pq_codes, centroids, 
        batch_neighbors, all_rel_pids, lambda_cut):
    loss = 0
    mrr = 0
    train_batch_size = len(batch_neighbors)
    for qembedding, retrieve_pids, cur_rel_pids in zip(
        query_embeddings, batch_neighbors, all_rel_pids):
        cur_rel_pids = list(set(cur_rel_pids))
        cur_rel_pids = torch.LongTensor(cur_rel_pids).to(batch_neighbors.device)
        target_labels = (retrieve_pids[:, None]==cur_rel_pids).any(-1)
        retrieved_rel_pids = retrieve_pids[target_labels]
        if len(retrieved_rel_pids) < len(cur_rel_pids):
            not_retrieved_rel_pids = cur_rel_pids[
                (cur_rel_pids[:, None]!=retrieved_rel_pids).all(-1)]
            assert len(not_retrieved_rel_pids) > 0
            retrieve_pids = torch.hstack([retrieve_pids, not_retrieved_rel_pids])
            target_labels = torch.hstack([target_labels, 
                torch.tensor([True]*len(not_retrieved_rel_pids)).to(target_labels.device)])
        position_matrix = (1+torch.arange(len(target_labels))).to(target_labels.device)
        if len(retrieved_rel_pids) > 0:
            first_rel_pos = position_matrix[target_labels][0].item()
            if first_rel_pos <= 10:
                mrr += 1/first_rel_pos 
        psg_embeddings = get_doc_embeds(retrieve_pids, pq_codes, centroids)
        cur_top_scores = (qembedding.reshape(1, -1) * psg_embeddings).sum(-1)
        rel_scores = cur_top_scores[target_labels]
        irrel_scores = cur_top_scores[~target_labels]
        rel_poses = position_matrix[target_labels]
        rel_weights = 1/rel_poses
        irrel_poses = position_matrix[~target_labels]
        irrel_weights = 1/irrel_poses
        if lambda_cut > 0:
            rel_pos_masks = (rel_poses <= lambda_cut).float()
            rel_weights *= rel_pos_masks
            irrel_pos_masks = (irrel_poses <= lambda_cut).float()
            irrel_weights *= irrel_pos_masks
        # relnum, irrelnum
        pair_diff = irrel_scores.reshape(1, -1) - rel_scores.reshape(-1, 1)
        pair_weights = torch.abs(irrel_weights.reshape(1, -1) - rel_weights.reshape(-1, 1))
        cur_loss = torch.log(1 + torch.exp(pair_diff)) * pair_weights
        loss += cur_loss.mean()
    mrr /= train_batch_size
    loss /= train_batch_size
    return loss, mrr


def train(args, model, pq_codes, centroid_embeds, opq_transform, opq_index):
    """ Train the model """
    ivf_index = faiss.downcast_index(opq_index.index)
    if args.gpu_search:
        res = faiss.StandardGpuResources()
        res.setTempMemory(128*1024*1024)
        co = faiss.GpuClonerOptions()
        co.useFloat16 = len(pq_codes) >= 56
        gpu_index = faiss.index_cpu_to_gpu(res, 0, opq_index, co)

    tb_writer = SummaryWriter(args.log_dir)

    train_dataset = TrainQueryDataset(
        TextTokenIdsCache(args.preprocess_dir, "train-query"),
        os.path.join(args.preprocess_dir, "train-qrel.tsv"),
        args.max_seq_length
    )

    train_sampler = RandomSampler(train_dataset) 
    collate_fn = get_collate_function(args.max_seq_length)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, 
        batch_size=args.train_batch_size, collate_fn=collate_fn)

    t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0},
        {'params': [centroid_embeds], 'weight_decay': args.centroid_weight_decay, 'lr': args.centroid_lr},
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.lr, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps,
        num_training_steps=t_total)
    
    # multi-gpu training (should be after apex fp16 initialization)
    # if args.n_gpu > 1:
    #     model = torch.nn.DataParallel(model)

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Total train batch size (w. accumulation) = %d",
                   args.train_batch_size * args.gradient_accumulation_steps)
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 0
    tr_loss, logging_loss = 0.0, 0.0
    tr_mrr, logging_mrr = 0.0, 0.0
    optimizer.zero_grad()
    train_iterator = trange(int(args.num_train_epochs), desc="Epoch")
    set_seed(args)  # Added here for reproductibility (even between python 2 and 3)  
    
    for epoch_idx, _ in enumerate(train_iterator):
        epoch_iterator = tqdm(train_dataloader, desc="Iteration")
        for step, (batch, all_rel_poffsets) in enumerate(epoch_iterator):

            batch = {k:v.to(args.model_device) for k, v in batch.items()}
            model.train()            
            query_embeddings = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"], 
                is_query=True)
            
            if args.gpu_search:
                batch_neighbors = gpu_index.search(
                    query_embeddings.detach().cpu().numpy(), args.loss_neg_topK)[1]
            else:
                batch_neighbors = opq_index.search(
                    query_embeddings.detach().cpu().numpy(), args.loss_neg_topK)[1]
            batch_neighbors = torch.tensor(batch_neighbors).to(model.device)
            loss, mrr = compute_loss(
                    query_embeddings @ opq_transform.T, 
                    pq_codes, centroid_embeds,
                    batch_neighbors, all_rel_poffsets, args.lambda_cut)   
            tr_mrr += mrr
            loss /= args.gradient_accumulation_steps
            loss.backward()

            tr_loss += loss.item()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            # if args.n_gpu > 1:
            #     loss = loss.mean() # mean() to average on multi-gpu parallel (not distributed) training
            if (step + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                optimizer.zero_grad()
                # model.zero_grad()
                global_step += 1
                faiss.copy_array_to_vector(
                    centroid_embeds.detach().cpu().numpy().ravel(), 
                    ivf_index.pq.centroids)
                if args.gpu_search:
                    gpu_index = None
                    gpu_index = faiss.index_cpu_to_gpu(res, 0, opq_index, co)
                if args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    tb_writer.add_scalar('lr', scheduler.get_lr()[0], global_step)
                    cur_loss =  (tr_loss - logging_loss)/args.logging_steps
                    tb_writer.add_scalar('train/all_loss', cur_loss, global_step)
                    logging_loss = tr_loss

                    cur_mrr =  (tr_mrr - logging_mrr)/(
                        args.logging_steps * args.gradient_accumulation_steps)
                    tb_writer.add_scalar('train/mrr_10', cur_mrr, global_step)
                    logging_mrr = tr_mrr
        
        save_model(model, args.model_save_dir, f'epoch-{epoch_idx+1}', args)
        faiss.write_index(opq_index,
            os.path.join(args.model_save_dir, f'epoch-{epoch_idx+1}', 
                os.path.basename(args.init_index_path)))


def run_parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--preprocess_dir", type=str, required=True)
    parser.add_argument("--model_save_dir", type=str, required=True)
    parser.add_argument("--log_dir", type=str, required=True)
    parser.add_argument("--init_index_path", type=str, required=True)
    parser.add_argument("--init_model_path", type=str, required=True)
    
    parser.add_argument("--lambda_cut", type=int, required=True)
    parser.add_argument("--centroid_lr", type=float, required=True)
    parser.add_argument("--gpu_search", action="store_true")
    parser.add_argument("--centroid_weight_decay", type=float, default=0)

    parser.add_argument("--loss_neg_topK", type=int, default=200)
    parser.add_argument("--max_seq_length", type=int, default=32)
    parser.add_argument("--train_batch_size", type=int, default=32)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--warmup_steps", default=2000, type=int)
    parser.add_argument("--no_cuda", action='store_true')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument("--logging_steps", type=int, default=100)

    parser.add_argument("--lr", default=5e-6, type=float)
    parser.add_argument("--weight_decay", default=0.01, type=float)
    parser.add_argument("--adam_epsilon", default=1e-8, type=float)
    parser.add_argument("--max_grad_norm", default=1.0, type=float)
    parser.add_argument("--num_train_epochs", default=6, type=int)

    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args()
    faiss.omp_set_num_threads(args.threads)

    os.makedirs(args.model_save_dir, exist_ok=True)
    return args


def main():
    args = run_parse_args()
    logger.info(args)

    # Setup CUDA, GPU 
    args.model_device = torch.device(f"cuda:0")
    args.n_gpu = 1

    logger.warning("Model Device: %s, n_gpu: %s", args.model_device, args.n_gpu)

    # Set seed
    set_seed(args)

    config = RobertaConfig.from_pretrained(args.init_model_path)
    config.return_dict = False
    config.gradient_checkpointing = args.gpu_search # to save cuda memory
    model = RobertaDot.from_pretrained(args.init_model_path, config=config)
    model.to(args.model_device)

    opq_index = faiss.read_index(args.init_index_path)
    vt = faiss.downcast_VectorTransform(opq_index.chain.at(0))            
    assert isinstance(vt, faiss.LinearTransform)
    opq_transform = faiss.vector_to_array(vt.A).reshape(vt.d_out, vt.d_in)
    opq_transform = torch.FloatTensor(opq_transform).to(args.model_device)

    ivf_index = faiss.downcast_index(opq_index.index)
    invlists = faiss.extract_index_ivf(ivf_index).invlists
    ls = invlists.list_size(0)
    pq_codes = faiss.rev_swig_ptr(invlists.get_codes(0), ls * invlists.code_size)
    pq_codes = pq_codes.reshape(-1, invlists.code_size)
    pq_codes = torch.LongTensor(pq_codes).to(args.model_device)

    centroid_embeds = faiss.vector_to_array(ivf_index.pq.centroids)
    centroid_embeds = centroid_embeds.reshape(ivf_index.pq.M, ivf_index.pq.ksub, ivf_index.pq.dsub)
    coarse_quantizer = faiss.downcast_index(ivf_index.quantizer)
    coarse_embeds = faiss.vector_to_array(coarse_quantizer.xb)
    centroid_embeds += coarse_embeds.reshape(ivf_index.pq.M, -1, ivf_index.pq.dsub)
    faiss.copy_array_to_vector(centroid_embeds.ravel(), ivf_index.pq.centroids)
    coarse_embeds[:] = 0   
    faiss.copy_array_to_vector(coarse_embeds.ravel(), coarse_quantizer.xb)

    centroid_embeds = torch.FloatTensor(centroid_embeds).to(args.model_device)
    centroid_embeds.requires_grad = True
    
    train(args, model, pq_codes, centroid_embeds, opq_transform, opq_index)
    

if __name__ == "__main__":
    main()
