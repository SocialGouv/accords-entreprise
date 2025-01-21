#!/usr/bin/env python
import argparse
import json
import os

from llama_index.core import SimpleDirectoryReader
from llama_index.core.node_parser import SentenceSplitter
from llama_index.finetuning import (
    SentenceTransformersFinetuneEngine,
    generate_qa_embedding_pairs,
)
from llama_index.finetuning.embeddings.common import EmbeddingQAFinetuneDataset
from llama_index.llms.openai import OpenAI

from tca.constants import FT_DATA_FOLDER

TRAIN_FILES = [f"./{FT_DATA_FOLDER}/10k/lyft_2021.pdf"]
VAL_FILES = [f"./{FT_DATA_FOLDER}/10k/uber_2021.pdf"]

TRAIN_CORPUS_FPATH = f"./{FT_DATA_FOLDER}/fake/train_corpus.json"
VAL_CORPUS_FPATH = f"./{FT_DATA_FOLDER}/fake/val_corpus.json"


def load_config(config_path):
    with open(config_path, "r") as f:
        return json.load(f)


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune embedding model")
    parser.add_argument(
        "--config", type=str, required=True, help="Path to the config file"
    )
    return parser.parse_args()


# TODO: Generate the training and validation datasets


def load_corpus(files, verbose=False):
    if verbose:
        print(f"Loading files {files}")

    reader = SimpleDirectoryReader(input_files=files)
    docs = reader.load_data()
    if verbose:
        print(f"Loaded {len(docs)} docs")

    parser = SentenceSplitter()
    nodes = parser.get_nodes_from_documents(docs, show_progress=verbose)

    if verbose:
        print(f"Parsed {len(nodes)} nodes")

    return nodes


def main(config) -> None:
    train_nodes = load_corpus(TRAIN_FILES, verbose=True)
    val_nodes = load_corpus(VAL_FILES, verbose=True)

    if os.path.exists(TRAIN_CORPUS_FPATH):
        train_dataset = EmbeddingQAFinetuneDataset.from_json(TRAIN_CORPUS_FPATH)
    else:
        train_dataset = generate_qa_embedding_pairs(
            llm=OpenAI(model="gpt-3.5-turbo"),
            nodes=train_nodes,  # type: ignore
            output_path=TRAIN_CORPUS_FPATH,
        )

    if os.path.exists(VAL_CORPUS_FPATH):
        val_dataset = EmbeddingQAFinetuneDataset.from_json(VAL_CORPUS_FPATH)
    else:
        val_dataset = generate_qa_embedding_pairs(
            llm=OpenAI(model="gpt-3.5-turbo"),
            nodes=val_nodes,  # type: ignore
            output_path=VAL_CORPUS_FPATH,
        )

    finetune_engine = SentenceTransformersFinetuneEngine(
        train_dataset,
        model_id=config["model_id"],
        model_output_path=config["model_output_path"],
        val_dataset=val_dataset,
        epochs=config["epochs"],
        batch_size=config["batch_size"],
        show_progress_bar=True,
        evaluation_steps=config["evaluation_steps"],
        trust_remote_code=True,
    )
    finetune_engine.finetune()
    embed_model = finetune_engine.get_finetuned_model()
    print(f"embed_model {embed_model}")


if __name__ == "__main__":
    args = parse_args()
    config = load_config(args.config)
    main(config)
