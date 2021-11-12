import os
import torch
import faiss
import logging
import numpy as np
from torch import nn
from numpy import ndarray
from torch import nn, Tensor
from tqdm.autonotebook import trange
from typing import List, Dict, Union, Tuple

from transformers import RobertaModel
from transformers.models.roberta.modeling_roberta import RobertaPreTrainedModel

from jpq.dataset import pack_tensor_2D
from jpq.star_tokenizer import RobertaTokenizer as JPQTokenizer

logger = logging.getLogger(__name__)


class RobertaDot(RobertaPreTrainedModel):
    def __init__(self, config):
        RobertaPreTrainedModel.__init__(self, config)
        self.roberta = RobertaModel(config, add_pooling_layer=False)
        self.embeddingHead = nn.Linear(config.hidden_size, config.hidden_size)
        self.norm = nn.LayerNorm(config.hidden_size)
        self.apply(self._init_weights)               
    
    def forward(self, input_ids, attention_mask):
        outputs1 = self.roberta(input_ids=input_ids,
                                attention_mask=attention_mask)
        full_emb = outputs1[0][:, 0]
        embeds = self.norm(self.embeddingHead(full_emb))
        return embeds


def batch_to_device(batch, target_device):
    """
    send a pytorch batch to a device (CPU/GPU)
    """
    for key in batch:
        if isinstance(batch[key], Tensor):
            batch[key] = batch[key].to(target_device)
    return batch


class JPQTower(RobertaDot):
    def __init__(self, config, max_input_length=512):
        super().__init__(config)
        assert config.hidden_size % config.MCQ_M == 0
        self.centroids = nn.Parameter(torch.zeros((
            config.MCQ_M, config.MCQ_K, config.hidden_size // config.MCQ_M)))
        self.rotation = nn.Parameter(torch.eye(config.hidden_size))
        self.tokenizer = JPQTokenizer.from_pretrained(
            "roberta-base", do_lower_case=True)
        self.max_input_length = max_input_length
    
    def forward(self, input_ids, attention_mask):
        unrotate_embeds = super().forward(input_ids, attention_mask)
        rotate_embeds = unrotate_embeds @ self.rotation.T
        return rotate_embeds

    def tokenize(self, texts: List[str]):
        texts = [t[:10000] if len(t) > 0 else " " for t in texts]
        features = self.tokenizer.batch_encode_plus(
            texts, max_length=self.max_input_length)
        features['input_ids'] = pack_tensor_2D(
                features['input_ids'], 
                default=self.tokenizer.pad_token_id, 
                dtype=torch.int64)
        features['attention_mask'] = pack_tensor_2D(
                features['attention_mask'], 
                default=0, 
                dtype=torch.int64)
        return features

    def encode(self, texts: Union[str, List[str]],
               batch_size: int = 32,
               show_progress_bar: bool = None,
               device: str = None, 
               index: faiss.Index = None) -> Union[ndarray, faiss.Index]:
        """
        Computes text embeddings

        :param texts: the texts to embed
        :param batch_size: the batch size used for the computation
        :param show_progress_bar: Output a progress bar when encode texts
        :param device: Which torch.device to use for the computation
        :param index: initial faiss index
        :return:
            Return index if index is given, else return numpy matrix
        """

        self.eval()
        if show_progress_bar is None:
            show_progress_bar = (logger.getEffectiveLevel()==logging.INFO or logger.getEffectiveLevel()==logging.DEBUG)

        input_was_string = False
        if isinstance(texts, str) or not hasattr(texts, '__len__'): #Cast an individual sentence to a list with length 1
            texts = [texts]
            input_was_string = True

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info("Use pytorch device: {}".format(device))

        self.to(device) # TODO: Utilize mutliple gpus

        all_embeddings = []

        for start_index in trange(0, len(texts), batch_size, desc="Batches", disable=not show_progress_bar):
            sentences_batch = texts[start_index:start_index+batch_size]
            features = self.tokenize(sentences_batch)
            features = batch_to_device(features, device)

            with torch.no_grad():
                embeddings = self.forward(**features)
                embeddings = embeddings.detach().cpu().numpy()

                if index is None:
                    all_embeddings.append(embeddings)
                else:
                    index.add(embeddings)

        if index is None:
            all_embeddings = np.vstack(all_embeddings)
            if input_was_string:
                all_embeddings = all_embeddings[0]
            return all_embeddings
        else:
            return index


class JPQDualEncoder:
    def __init__(self, model_path: Tuple, sep: str =". ", **kwargs):
        self.sep = sep
        self.q_model = JPQTower.from_pretrained(model_path[0])
        self.doc_model = JPQTower.from_pretrained(model_path[1])
    
    def encode_queries(self, queries: List[str], batch_size: int = 16, **kwargs) -> np.ndarray:
        return self.q_model.encode(queries, batch_size=batch_size, **kwargs)
    
    def encode_corpus(self, corpus: List[Dict[str, str]], batch_size: int = 8, faiss_metric = faiss.METRIC_INNER_PRODUCT, **kwargs) -> faiss.Index:
        # Init fake PQ index
        D, M = self.doc_model.config.hidden_size, self.doc_model.config.MCQ_M
        coarse_quantizer = faiss.IndexFlatL2(D)
        assert self.doc_model.config.MCQ_K == 256
        index = faiss.IndexIVFPQ(coarse_quantizer, D, 1, M, 8, faiss_metric)
        fake_train_pts = np.random.random((10000, D)).astype(np.float32)
        index.train(fake_train_pts) # fake training

        # ignore coarse quantizer
        coarse_quantizer = faiss.downcast_index(index.quantizer)
        coarse_embeds = faiss.vector_to_array(coarse_quantizer.xb)
        coarse_embeds[:] = 0
        faiss.copy_array_to_vector(coarse_embeds.ravel(), coarse_quantizer.xb)
        # set centroid values 
        doc_centroids = self.doc_model.centroids.data.detach().cpu().numpy()
        faiss.copy_array_to_vector(doc_centroids.ravel(), index.pq.centroids)
        # some other stuffs
        index.precomputed_table.clear()
        index.precompute_table()

        # encode documents and add to index 
        sentences = [(doc["title"] + self.sep + doc["text"]).strip() if "title" in doc else doc["text"].strip() for doc in corpus]
        index = self.doc_model.encode(sentences, batch_size=batch_size, index = index, **kwargs)
        
        # re-set centroid embeddings
        query_centroids = self.q_model.centroids.data.detach().cpu().numpy()
        faiss.copy_array_to_vector(query_centroids.ravel(), index.pq.centroids)
        return index


class DenseRetrievalJPQSearch:
    
    def __init__(self, model, batch_size: int = 128, corpus_index: faiss.Index = None, **kwargs):
        #model is class that provides encode_corpus() and encode_queries()
        self.model = model
        self.batch_size = batch_size
        # Faiss has no cosine similarity metric
        self.score_functions = {'dot': faiss.METRIC_INNER_PRODUCT}
        self.score_function_desc = {'dot': "Dot Product"}
        # Since we use compact Index, this is not required
        # self.corpus_chunk_size = corpus_chunk_size 
        self.show_progress_bar = True #TODO: implement no progress bar if false
        # self.convert_to_tensor = True : Faiss uses numpy

        # so we can reuse stored faiss index
        # and do not have to encode the corpus again
        self.corpus_index = corpus_index
        self.results = {}
    
    def search(self, 
               corpus: Dict[str, Dict[str, str]], 
               queries: Dict[str, str], 
               top_k: int, 
               score_function: str,
                **kwargs) -> Dict[str, Dict[str, float]]:
        #Create embeddings for all queries using model.encode_queries()
        #Runs semantic search against the corpus embeddings
        #Returns a ranked list with the corpus ids
        if score_function not in self.score_functions:
            raise ValueError("score function: {} must be either (dot) for dot product".format(score_function))
            
        logger.info("Encoding Queries...")
        query_ids = list(queries.keys())
        self.results = {qid: {} for qid in query_ids}
        queries = [queries[qid] for qid in queries]
        query_embeddings = self.model.encode_queries(
            queries, 
            batch_size=self.batch_size, 
            show_progress_bar=self.show_progress_bar)
          
        logger.info("Sorting Corpus by document length (Longest first)...")
        corpus_ids = sorted(corpus, key=lambda k: len(corpus[k].get("title", "") + corpus[k].get("text", "")), reverse=True)
        corpus = [corpus[cid] for cid in corpus_ids]

        if self.corpus_index is None:
            logger.info("Encoding Corpus in batches... Warning: This might take a while!")
            logger.info("Scoring Function: {} ({})".format(self.score_function_desc[score_function], score_function))
            self.corpus_index = self.model.encode_corpus(
                corpus, batch_size=self.batch_size,
                faiss_metric=self.score_functions[score_function],
                show_progress_bar=self.show_progress_bar
            )
        else:
            logger.warning("Skip the corpus encoding process and utilize pre-computed corpus_index")
        
        # keep self.corpus_index on cpu
        if faiss.get_num_gpus() == 1:
            logger.info("Transfering index to GPU-0")
            res = faiss.StandardGpuResources()
            co = faiss.GpuClonerOptions()
            co.useFloat16 = faiss.downcast_index(self.corpus_index).pq.M >= 56
            corpus_index = faiss.index_cpu_to_gpu(res, 0, self.corpus_index, co)
        elif faiss.get_num_gpus() > 1:
            logger.info("Transfering index to multiple GPUs")
            co = faiss.GpuMultipleClonerOptions()
            co.useFloat16 = faiss.downcast_index(self.corpus_index).pq.M >= 56
            corpus_index = faiss.index_cpu_to_all_gpus(self.corpus_index, co)
        else:
            corpus_index = self.corpus_index

        logger.info("Begin search")
        top_k_values, top_k_idx = corpus_index.search(query_embeddings, top_k+1)

        logger.info("Writing results")
        for query_itr in range(len(query_embeddings)):
            query_id = query_ids[query_itr]                  
            for corpus_id_offset, score in zip(top_k_idx[query_itr], top_k_values[query_itr]):
                corpus_id = corpus_ids[corpus_id_offset]
                # Ignore self and empty text
                if corpus_id != query_id and len(corpus[corpus_id_offset]['text'].strip()) > 0:
                    self.results[query_id][corpus_id] = float(score)
        return self.results 


