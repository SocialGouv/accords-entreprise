#!/usr/bin/env python
from pathlib import Path

from llama_index.finetuning.embeddings.common import EmbeddingQAFinetuneDataset
from sentence_transformers import SentenceTransformer
from sentence_transformers.evaluation import InformationRetrievalEvaluator

from tca.constants import FT_DATA_FOLDER


def evaluate_st(
    dataset: EmbeddingQAFinetuneDataset,
    model_id: str,
    name: str,
) -> dict[str, float]:
    corpus = dataset.corpus
    queries = dataset.queries
    relevant_docs = dataset.relevant_docs

    evaluator = InformationRetrievalEvaluator(queries, corpus, relevant_docs, name=name)  # type: ignore
    model = SentenceTransformer(model_id)
    output_path = "results/"
    Path(output_path).mkdir(exist_ok=True, parents=True)
    return evaluator(model, output_path=output_path)


def eval_bge(val_dataset: EmbeddingQAFinetuneDataset) -> None:
    result = evaluate_st(val_dataset, "BAAI/bge-small-en", name="bge")
    print(f"result {result}")


def eval_fine_tuned(val_dataset: EmbeddingQAFinetuneDataset) -> None:
    result = evaluate_st(val_dataset, "checkpoints/final", name="finetuned")
    print(f"result {result}")


if __name__ == "__main__":
    val_dataset = EmbeddingQAFinetuneDataset.from_json(
        f"{FT_DATA_FOLDER}/val_dataset.json"
    )

    # eval_bge(val_dataset)
    eval_fine_tuned(val_dataset)
