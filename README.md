# Assigner des thèmes aux accords d'entreprise

## Docker PostgreSQL

```bash
docker-compose up -d
docker exec -it postgres-pgvector psql -U $POSTGRES_USER -d $POSTGRES_DB -c "\d"
```

## Alembic

### Init Alembic

```bash
# To update DB schemas
alembic revision --autogenerate -m "Initial migration"
# Open up the generated revision and fix imports that might be wrong (because of pgvector for example)

# To apply DB schemas
alembic upgrade head
```

### Reset the DB and alembic entirely

```bash
# Delete all alembic versions
rm -rf tca/alembic/versions/*.py

# Drop the database
docker exec -it postgres-pgvector psql -U $POSTGRES_USER -d postgres -c "DROP DATABASE $POSTGRES_DB;"

# Create the database again
docker exec -it postgres-pgvector psql -U $POSTGRES_USER -d postgres -c "CREATE DATABASE $POSTGRES_DB;"

# Run `scripts/init.sql` against the PostgreSQL DB in the Docker container:
docker exec -i postgres-pgvector psql -U $POSTGRES_USER -d $POSTGRES_DB -f docker-entrypoint-initdb.d/init.sql
```

Then re-run [Init Alembic](#init-alembic)

## Embeddings

### Using Ollama locally

```bash
ollama pull bge-m3:567m-fp16
```

## Scripts

### Ingest documents

```bash
make ingest-docs
```

### Ingest themes

```bash
make ingest-themes
```

### Theming

```bash
make run-theming
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
