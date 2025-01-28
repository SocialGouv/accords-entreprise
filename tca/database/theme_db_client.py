#!/usr/bin/env python
import logging
import logging.config
import time
from typing import TypedDict

import pandas as pd
from sqlalchemy.orm import Session

from tca.custom_types import Embeddings
from tca.database.models import (
    BaseThemeEmbedding,
    Theme,
)
from tca.embedding.embedding_clients import BaseEmbeddingClient

logging.config.fileConfig("logging.conf")


class ThemeWithEmbedding(TypedDict):
    theme: Theme
    embeddings: Embeddings


class ThemeDBClient:
    def __init__(
        self,
        session: Session,
        theme_embedding_cls: type[BaseThemeEmbedding],
    ) -> None:
        self.session = session
        self.db_theme_prompt_embedding_cls = theme_embedding_cls

    def ingest_themes_in_db(
        self, embedding_client: BaseEmbeddingClient, themes: pd.DataFrame
    ) -> None:
        logging.info("Ingesting themes in the database")
        prompts = themes["description"].tolist()
        all_theme_prompt_embeddings = embedding_client.encode_queries(prompts)

        for idx in range(len(themes)):
            theme = themes.iloc[idx]
            prompt = prompts[idx]
            theme_prompt_embeddings = all_theme_prompt_embeddings[idx]

            current_timestamp = int(time.time())
            built_theme = Theme(
                prompt=prompt,
                description=theme["description"],
                themes=[theme["niveau 1"], theme["niveau 2"]],
                created_at=current_timestamp,
                updated_at=current_timestamp,
            )
            logging.info(f"Adding theme: {theme['niveau 2']}")
            self.session.add(built_theme)
            self.session.flush()
            theme_embedding = self.db_theme_prompt_embedding_cls(
                theme_id=built_theme.id,
                embeddings=theme_prompt_embeddings,
                created_at=current_timestamp,
                updated_at=current_timestamp,
            )
            self.session.add(theme_embedding)
        self.session.commit()

    def get_themes_with_their_embeddings(
        self,
    ) -> list[ThemeWithEmbedding]:
        # I am not doing this in a single query because sqlalchemy does not support
        # de-duplicating numpy arrays and it needs to do that if I join these two tables...
        themes = self.session.query(Theme).all()

        embeddings_result = self.session.query(
            self.db_theme_prompt_embedding_cls.theme_id,
            self.db_theme_prompt_embedding_cls.embeddings,
        ).all()
        embeddings_dict = {
            embedding.theme_id: embedding.embeddings for embedding in embeddings_result
        }

        return [
            ThemeWithEmbedding(theme=theme, embeddings=embeddings_dict[theme.id])
            for theme in themes
        ]
