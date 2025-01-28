#!/usr/bin/env python
import logging
import logging.config
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score

logging.config.fileConfig("logging.conf")


class ThemeProcessor:
    @staticmethod
    def load_themes(theme_list_path: Path) -> pd.DataFrame:
        return pd.read_csv(theme_list_path, encoding="UTF8").map(
            lambda s: s.lower() if isinstance(s, str) else s
        )

    @staticmethod
    def evaluate_classification(
        theme_assignment_df: pd.DataFrame, expected_df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Evaluates theme classification performance using standard metrics.

        Args:
            theme_assignment_df: DataFrame with predicted theme assignments
            expected_df: DataFrame with ground truth theme assignments

        Returns:
            Tuple of (detailed metrics DataFrame, overall metrics dict)
        """
        # Create binary indicators for each theme combination
        all_docs = pd.concat([theme_assignment_df, expected_df])["Document"].unique()

        per_theme_metrics_data = []
        for theme1 in expected_df["Thème n1"].unique():
            for theme2 in np.unique(
                expected_df[expected_df["Thème n1"] == theme1]["Thème n2"]
            ):
                # Create binary vectors for this theme combination
                y_true = np.zeros(len(all_docs))
                y_pred = np.zeros(len(all_docs))

                true_docs = set(
                    expected_df[
                        (expected_df["Thème n1"] == theme1)
                        & (expected_df["Thème n2"] == theme2)
                    ]["Document"]
                )

                pred_docs = set(
                    theme_assignment_df[
                        (theme_assignment_df["Thème n1"] == theme1)
                        & (theme_assignment_df["Thème n2"] == theme2)
                    ]["Document"]
                )

                for i, doc_id in enumerate(all_docs):
                    y_true[i] = 1 if doc_id in true_docs else 0
                    y_pred[i] = 1 if doc_id in pred_docs else 0

                # Calculate metrics
                tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

                per_theme_metrics_data.append(
                    {
                        "Thème n1": theme1,
                        "Thème n2": theme2,
                        "Precision": precision_score(y_true, y_pred, zero_division=0),  # type: ignore
                        "Recall": recall_score(y_true, y_pred, zero_division=0),  # type: ignore
                        "F1": f1_score(y_true, y_pred, zero_division=0),  # type: ignore
                        "True Positives": tp,
                        "False Positives": fp,
                        "False Negatives": fn,
                        "Support": len(true_docs),
                    }
                )

        # Create detailed metrics DataFrame
        per_theme_metrics_df = pd.DataFrame(per_theme_metrics_data)
        per_theme_metrics_df.sort_values(["Thème n1", "Thème n2"], inplace=True)

        # Calculate macro averages
        overall_metrics_df = pd.DataFrame(
            {
                "Macro Precision": [per_theme_metrics_df["Precision"].mean()],
                "Macro Recall": [per_theme_metrics_df["Recall"].mean()],
                "Macro F1": [per_theme_metrics_df["F1"].mean()],
                "Total Support": [per_theme_metrics_df["Support"].sum()],
            }
        )

        # Format percentages
        for col in ["Precision", "Recall", "F1"]:
            per_theme_metrics_df[col] = (per_theme_metrics_df[col] * 100).round(1)

        return per_theme_metrics_df, overall_metrics_df

    @staticmethod
    def save_theming_results(
        per_theme_metrics_df: pd.DataFrame,
        overall_metrics_df: pd.DataFrame,
        output_path: str,
    ) -> None:
        with pd.ExcelWriter(output_path, engine="xlsxwriter") as writer:
            header_format = writer.book.add_format({"bold": True})  # type: ignore

            per_theme_metrics_df.to_excel(
                writer, index=False, sheet_name="Theme Results"
            )
            worksheet = writer.sheets["Theme Results"]
            worksheet.freeze_panes(1, 0)
            for col_num, value in enumerate(per_theme_metrics_df.columns.values):
                worksheet.write(0, col_num, value, header_format)
            worksheet.set_zoom(140)

            overall_metrics_df.to_excel(
                writer, index=False, sheet_name="Overall Results"
            )
            worksheet_found_ratio = writer.sheets["Overall Results"]
            worksheet_found_ratio.freeze_panes(1, 0)
            for col_num, value in enumerate(overall_metrics_df.columns.values):
                worksheet_found_ratio.write(0, col_num, value, header_format)
            worksheet_found_ratio.set_zoom(140)
