#!/usr/bin/env python
import logging
import logging.config
import time

import pandas as pd
from sqlalchemy import Row
from sqlalchemy.orm import Session

from tca.custom_types import Embeddings
from tca.database.models import (
    BaseThemeEmbedding,
    Theme,
)
from tca.embedding.embedding_clients import BaseEmbeddingClient

logging.config.fileConfig("logging.conf")
logger = logging.getLogger(__name__)


class ThemeDBClient:
    def __init__(
        self,
        session: Session,
        embedding_client: BaseEmbeddingClient,
        db_theme_prompt_embedding_cls: type[BaseThemeEmbedding],
    ) -> None:
        self.session = session
        self.embedding_client = embedding_client
        self.db_theme_prompt_embedding_cls = db_theme_prompt_embedding_cls

    def ingest_themes_in_db(self, themes: pd.DataFrame) -> None:
        for _index, theme in themes.iterrows():
            # prompt = f"{theme['niveau 1']} -> {theme['niveau 2']}"
            prompt = f'Dans le contexte des accords d\'entreprise français, trouvez les passages qui mentionnent exactement : "{theme["niveau 2"]}"'
            # prompt = (
            #     f"Dans le contexte des accords d'entreprise français et plus précisément dans le domaine '{theme['niveau 1']}', "
            #     f"trouvez les passages qui traitent exactement de : {theme['niveau 2']}"
            # )
            theme_prompt_embeddings = self.embedding_client.build_embedding([prompt])[0]
            current_timestamp = int(time.time())
            theme = Theme(
                prompt=prompt,
                themes=[theme["niveau 1"], theme["niveau 2"]],
                created_at=current_timestamp,
                updated_at=current_timestamp,
            )
            self.session.add(theme)
            self.session.flush()
            theme_embedding = self.db_theme_prompt_embedding_cls(
                theme_id=theme.id,
                embedding=theme_prompt_embeddings,
                created_at=current_timestamp,
                updated_at=current_timestamp,
            )
            self.session.add(theme_embedding)
        self.session.commit()

    def get_themes_with_embeddings(self) -> list[Row[tuple[list[str], Embeddings]]]:
        return (
            self.session.query(
                Theme.themes,
                self.db_theme_prompt_embedding_cls.embedding.label(
                    "theme_prompt_embedding"
                ),
            )
            .join(
                self.db_theme_prompt_embedding_cls,
                Theme.id == self.db_theme_prompt_embedding_cls.theme_id,
            )
            .all()
        )
