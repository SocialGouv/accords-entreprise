#!/usr/bin/env python
import logging
import logging.config
import os
import re

import pandas as pd

from tca.database.models import OllamaBgeM3Embedding
from tca.database.session_manager import PostgresSessionManager
from tca.document_chunk_manager import DocumentChunkManager, DocumentChunkManagerConfig
from tca.document_ingester import DATA_FOLDER
from tca.embedding.embedding_clients import OllamaEmbeddingClient

logging.config.fileConfig("logging.conf")
logger = logging.getLogger(__name__)

DATA_FOLDER = os.getenv("DATA_FOLDER", "data")


# TODO: This early test should show that the pipeline works. Store the result so it can be compared to our labeled data
# TODO: Finally, implement the real embedding and chunking logic using a good embedder and chunking with langchain or similar
def main() -> None:
    postgres_session_manager = PostgresSessionManager()
    session = postgres_session_manager.session

    embedding_client = OllamaEmbeddingClient()
    ollama_bge_m3_config = DocumentChunkManagerConfig(
        embedding_client=embedding_client,
        db_embedding_model_cls=OllamaBgeM3Embedding,
    )

    document_chunk_manager = DocumentChunkManager(
        session,
        ollama_bge_m3_config,
    )

    theme_list_path = os.path.join(DATA_FOLDER, "theme_list.csv")

    themes = pd.read_csv(theme_list_path)
    results = []
    for _index, theme in themes.iterrows():
        # prompt = (
        #     f"Generate an embedding for a theme specifically related to company agreements in France. "
        #     f"The theme hierarchy, from broader to narrower, is: {theme['niveau 1']} -> {theme['niveau 2']}. "
        #     f"Focus the embedding on representing the semantic meaning of this theme in the context of French labor law and company agreements."
        # )
        prompt = f"{theme['niveau 1']} -> {theme['niveau 2']}"

        query_embeddings = embedding_client.embed([prompt])[0]
        semantic_search_results = document_chunk_manager.query_similar_chunks(
            query_embeddings=query_embeddings, cos_dist_threshold=0.4, top_k=1
        )
        if semantic_search_results:
            results.append(
                {
                    "themes": [theme["niveau 1"], theme["niveau 2"]],
                    "semantic_search_results": semantic_search_results,
                }
            )

    output_data = []

    for result in results:
        for search_result in result["semantic_search_results"]:
            chunk = search_result["chunk"]
            document_id_match = re.match(r"T\d+", chunk.document_name)
            document_id = document_id_match.group(0) if document_id_match else "Unknown"
            output_data.append(
                {
                    "Document ID": document_id,
                    "Thème n1": result["themes"][0],
                    "Thème n2": result["themes"][1],
                    "Distance": search_result["distance"],
                    "Chunk": chunk.chunk_text,
                }
            )

    output_df = pd.DataFrame(output_data)

    expected_df = pd.read_excel(os.path.join(DATA_FOLDER, "normalized_themes.xlsx"))

    comparison_df = pd.merge(
        output_df,
        expected_df,
        on=["Document ID", "Thème n1", "Thème n2"],
        how="outer",
        indicator=True,
    )

    comparison_df["Found"] = comparison_df["_merge"] == "both"
    comparison_df.drop(columns=["_merge"], inplace=True)
    theme_performance = (
        comparison_df.groupby(["Thème n1", "Thème n2"])["Found"]
        .mean()
        .reset_index()
        .rename(columns={"Found": "Found Ratio"})
    )
    theme_performance["Found Ratio"] = (theme_performance["Found Ratio"] * 100).round(1)
    theme_performance.sort_values(by=["Thème n1", "Thème n2"], inplace=True)

    with pd.ExcelWriter(
        os.path.join(DATA_FOLDER, "comparison_results.xlsx"),
        engine="xlsxwriter",
    ) as writer:
        column_order = [
            "Document ID",
            "Thème n1",
            "Thème n2",
            "Distance",
            "Found",
            "Chunk",
        ]
        comparison_df.sort_values(
            by=["Document ID", "Thème n1", "Thème n2"], inplace=True
        )
        comparison_df[column_order].to_excel(
            writer, index=False, sheet_name="Comparison"
        )
        worksheet = writer.sheets["Comparison"]
        worksheet.freeze_panes(1, 0)

        # Set column widths
        worksheet.set_column("A:E", 20)
        worksheet.set_column("F:F", 50)

        header_format = writer.book.add_format({"bold": True})  # type: ignore
        for col_num, value in enumerate(comparison_df.columns.values):
            worksheet.write(0, col_num, value, header_format)

        # Set zoom to 140%
        worksheet.set_zoom(140)

        # Output the "Found Ratio" to a new sheet
        theme_performance.to_excel(writer, index=False, sheet_name="Found Ratio")
        worksheet_found_ratio = writer.sheets["Found Ratio"]
        worksheet_found_ratio.freeze_panes(1, 0)

        # Set column widths for the "Found Ratio" sheet
        worksheet_found_ratio.set_column("A:B", 20)
        worksheet_found_ratio.set_column("C:C", 15)

        for col_num, value in enumerate(theme_performance.columns.values):
            worksheet_found_ratio.write(0, col_num, value, header_format)


if __name__ == "__main__":
    main()
