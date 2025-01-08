#!/usr/bin/env python
import logging
import logging.config
import os
import re
from pathlib import Path
from typing import Any, Dict, Tuple

import pandas as pd

from tca.constants import DATA_FOLDER
from tca.database.models import OllamaBgeM3Embedding
from tca.database.session_manager import PostgresSessionManager
from tca.document_chunk_manager import DocumentChunkManager, DocumentChunkManagerConfig
from tca.embedding.embedding_clients import OllamaEmbeddingClient

logging.config.fileConfig("logging.conf")
logger = logging.getLogger(__name__)


class ThemeDocumentProcessor:
    def __init__(self) -> None:
        self.embedding_client = OllamaEmbeddingClient()
        self.document_chunk_manager = DocumentChunkManager(
            PostgresSessionManager().session,
            DocumentChunkManagerConfig(
                embedding_client=self.embedding_client,
                db_embedding_model_cls=OllamaBgeM3Embedding,
            ),
        )

    def load_themes(self, theme_list_path: Path) -> pd.DataFrame:
        return pd.read_csv(theme_list_path).map(
            lambda s: s.lower() if isinstance(s, str) else s
        )

    def assign_themes_to_docs(self, themes: pd.DataFrame) -> list[Dict[str, Any]]:
        results = []
        for _index, theme in themes.iterrows():
            # prompt = f"{theme['niveau 1']} -> {theme['niveau 2']}"
            # prompt = f"Dans le contexte des accords d'entreprise français, trouvez les passages qui traitent spécifiquement de : {theme['niveau 2']}"
            prompt = (
                f"Dans le contexte des accords d'entreprise français et plus précisément dans le domaine '{theme['niveau 1']}', "
                f"trouvez les passages qui traitent spécifiquement de : {theme['niveau 2']}"
            )
            query_embeddings = self.embedding_client.embed([prompt])[0]
            semantic_search_results = (
                self.document_chunk_manager.find_matching_documents(
                    query_embeddings=query_embeddings, cos_dist_threshold=0.4
                )
            )
            if semantic_search_results:
                results.append(
                    {
                        "themes": [theme["niveau 1"], theme["niveau 2"]],
                        "semantic_search_results": semantic_search_results,
                    }
                )
        return results

    def build_theme_assignment_df(
        self, theme_assignments: list[Dict[str, Any]]
    ) -> pd.DataFrame:
        structured_theme_assignment = []
        for theme_assignment in theme_assignments:
            for search_result in theme_assignment["semantic_search_results"]:
                chunk = search_result["chunk"]
                document_id_match = re.match(r"[AT]\d+", chunk.document_name)
                document_id = (
                    document_id_match.group(0) if document_id_match else "Unknown"
                )
                document_id = document_id.lower()
                structured_theme_assignment.append(
                    {
                        "Document ID": document_id,
                        "Thème n1": theme_assignment["themes"][0],
                        "Thème n2": theme_assignment["themes"][1],
                        "Distance": search_result["distance"],
                        "Chunk": chunk.chunk_text,
                    }
                )
        return pd.DataFrame(structured_theme_assignment)

    def compare_with_expected(
        self, theme_assignment_df: pd.DataFrame, expected_df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        comparison_df = pd.merge(
            theme_assignment_df,
            expected_df,
            on=["Document ID", "Thème n1", "Thème n2"],
            how="outer",
            indicator=True,
        )
        comparison_df["Found"] = comparison_df["_merge"] == "both"
        comparison_df.drop(columns=["_merge"], inplace=True)
        theme_performance_df = (
            comparison_df.groupby(["Thème n1", "Thème n2"])["Found"]
            .mean()
            .reset_index()
            .rename(columns={"Found": "Found Ratio"})
        )
        theme_performance_df["Found Ratio"] = (
            theme_performance_df["Found Ratio"] * 100
        ).round(1)
        theme_performance_df.sort_values(by=["Thème n1", "Thème n2"], inplace=True)
        return comparison_df, theme_performance_df

    def save_theming_results(
        self,
        comparison_df: pd.DataFrame,
        theme_performance_df: pd.DataFrame,
        output_path: str,
    ) -> None:
        with pd.ExcelWriter(output_path, engine="xlsxwriter") as writer:
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
            theme_performance_df.sort_values(by=["Found Ratio"], inplace=True)
            theme_performance_df.to_excel(writer, index=False, sheet_name="Found Ratio")
            worksheet_found_ratio = writer.sheets["Found Ratio"]
            worksheet_found_ratio.freeze_panes(1, 0)

            # Set column widths for the "Found Ratio" sheet
            worksheet_found_ratio.set_column("A:B", 20)
            worksheet_found_ratio.set_column("C:C", 15)

            for col_num, value in enumerate(theme_performance_df.columns.values):
                worksheet_found_ratio.write(0, col_num, value, header_format)
            worksheet_found_ratio.set_zoom(140)


def main() -> None:
    processor = ThemeDocumentProcessor()
    theme_list_path = Path(os.path.join(DATA_FOLDER, "theme_list.csv"))
    themes = processor.load_themes(theme_list_path)
    results = processor.assign_themes_to_docs(themes)
    output_df = processor.build_theme_assignment_df(results)
    expected_df_path = Path(os.path.join(DATA_FOLDER, "normalized_themes.xlsx"))
    expected_df = pd.read_excel(expected_df_path).map(
        lambda s: s.lower() if isinstance(s, str) else s
    )

    comparison_df, theme_performance_df = processor.compare_with_expected(
        output_df, expected_df
    )
    output_path = os.path.join(DATA_FOLDER, "comparison_results.xlsx")
    processor.save_theming_results(comparison_df, theme_performance_df, output_path)


if __name__ == "__main__":
    main()
