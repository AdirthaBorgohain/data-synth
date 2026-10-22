FROM python:3.12-slim

WORKDIR /app

COPY pyproject.toml generate.py ./

RUN pip install --no-cache-dir \
    "faker>=30.0" \
    "numpy>=2.0" \
    "psycopg[binary]>=3.2"

CMD ["python", "generate.py"]
