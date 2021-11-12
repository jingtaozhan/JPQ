import sys
sys.path.append("./")
import argparse
import logging
import random, os
import pickle, faiss
from beir import util, LoggingHandler
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval

from jpq.model import JPQDualEncoder
from jpq.model import DenseRetrievalJPQSearch as DRJS

#### Just some code to print debug information to stdout
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])
#### /print debug information to stdout


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--beir_data_root", type=str, required=True)
    parser.add_argument("--query_encoder", type=str, required=True)
    parser.add_argument("--doc_encoder", type=str, required=True)
    parser.add_argument("--split", type=str, default='test')
    parser.add_argument("--encode_batch_size", type=int, default=64)
    parser.add_argument("--output_index_path", type=str, default=None)
    parser.add_argument("--output_ranking_path", type=str, default=None)
    args = parser.parse_args()

    #### Download scifact.zip dataset and unzip the dataset
    dataset = args.dataset
    url = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{}.zip".format(dataset)
    data_path = util.download_and_unzip(url, args.beir_data_root)

    #### Provide the data_path where scifact has been downloaded and unzipped
    corpus, queries, qrels = GenericDataLoader(data_folder=data_path).load(split=args.split)

    #### Load pre-computed index
    if args.output_index_path is not None and os.path.isfile(args.output_index_path):
        corpus_index = faiss.read_index(args.output_index_path)
    else:
        corpus_index = None

    #### Load the RepCONC model and retrieve using dot-similarity
    model = DRJS(JPQDualEncoder((args.query_encoder, args.doc_encoder),), batch_size=args.encode_batch_size, corpus_index=corpus_index)
    retriever = EvaluateRetrieval(model, score_function="dot") # or "dot" for dot-product
    results = retriever.retrieve(corpus, queries)

    #### Evaluate your model with NDCG@k, MAP@K, Recall@K and Precision@K  where k = [1,3,5,10,100,1000] 
    ndcg, _map, recall, precision = retriever.evaluate(qrels, results, retriever.k_values)

    if args.output_index_path is not None:
        os.makedirs(os.path.dirname(args.output_index_path), exist_ok=True)
        faiss.write_index(model.corpus_index, args.output_index_path)

    if args.output_ranking_path is not None:
        os.makedirs(os.path.dirname(args.output_ranking_path), exist_ok=True)
        pickle.dump(results, open(args.output_ranking_path, 'wb'))
