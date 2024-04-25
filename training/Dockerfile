FROM python:3.11-slim-bullseye

# RUN mkdir api
WORKDIR  /app

COPY /pyproject.toml .
COPY /poetry.lock .

RUN pip3 install poetry
RUN poetry config virtualenvs.create false
RUN poetry install

COPY . app/

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]