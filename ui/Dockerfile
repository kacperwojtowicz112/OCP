FROM python:3.11-slim-bullseye


COPY /pyproject.toml .
COPY /poetry.lock .

RUN pip3 install poetry
RUN poetry config virtualenvs.create false
RUN poetry install
WORKDIR /app
COPY . /app

EXPOSE 8501

ENTRYPOINT ["streamlit","run"]

CMD ["app.py"]