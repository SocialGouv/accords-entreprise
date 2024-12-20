# Assigner des thèmes aux accords d'entreprise

## Alembic

```bash
# To update DB schemas
alembic revision --autogenerate -m "A message"
# To apply DB schemas
alembic upgrade head
```

## Docker PostgreSQL

```bash
docker-compose up -d
docker exec -it postgres-pgvector psql -U user -d company_agreements -c "\d"
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
