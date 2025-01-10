#!/usr/bin/env python
import logging
import logging.config
import re
from pathlib import Path
from typing import Any, Dict, Tuple

import pandas as pd

from tca.embedding.embedding_clients import BaseEmbeddingClient

logging.config.fileConfig("logging.conf")
logger = logging.getLogger(__name__)


class ThemeProcessor:
    def __init__(
        self,
        embedding_client: BaseEmbeddingClient,
    ) -> None:
        self.embedding_client = embedding_client

    def load_themes(self, theme_list_path: Path) -> pd.DataFrame:
        return pd.read_csv(theme_list_path).map(
            lambda s: s.lower() if isinstance(s, str) else s
        )

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
        for status, column_name in [
            ("both", "Found"),
            ("right_only", "Incorrectly Found"),
        ]:
            comparison_df[column_name] = comparison_df["_merge"] == status
            performance_df = (
                comparison_df.groupby(["Thème n1", "Thème n2"])[column_name]
                .mean()
                .reset_index()
                .rename(columns={column_name: f"{column_name} Ratio"})
            )
            performance_df[f"{column_name} Ratio"] = (
                performance_df[f"{column_name} Ratio"] * 100
            ).round(1)
            performance_df.sort_values(by=["Thème n1", "Thème n2"], inplace=True)
            if column_name == "Found":
                theme_performance_df = performance_df
            else:
                incorrect_theme_performance_df = performance_df

        comparison_df.drop(columns=["_merge"], inplace=True)

        # Merge the correct and incorrect theme performance dataframes
        theme_performance_df = pd.merge(
            theme_performance_df,
            incorrect_theme_performance_df,
            on=["Thème n1", "Thème n2"],
            how="left",
        )
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
            theme_performance_df.sort_values(
                by=["Found Ratio", "Incorrectly Found Ratio"], inplace=True
            )
            theme_performance_df.to_excel(writer, index=False, sheet_name="Analysis")
            worksheet_found_ratio = writer.sheets["Analysis"]
            worksheet_found_ratio.freeze_panes(1, 0)

            # Set column widths for the "Analysis" sheet
            worksheet_found_ratio.set_column("A:B", 20)
            worksheet_found_ratio.set_column("C:D", 15)

            for col_num, value in enumerate(theme_performance_df.columns.values):
                worksheet_found_ratio.write(0, col_num, value, header_format)
            worksheet_found_ratio.set_zoom(140)
