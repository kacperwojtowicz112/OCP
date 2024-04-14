FROM python:3.11

RUN mkdir api
WORKDIR  /api

COPY /pyproject.toml .
COPY /poetry.lock .

RUN pip3 install poetry
RUN poetry config virtualenvs.create false
RUN poetry install

COPY . .

EXPOSE 8000

CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]