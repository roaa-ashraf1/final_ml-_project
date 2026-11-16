FROM python:3.11-slim

WORKDIR /code

COPY requirements.txt /code/requirements.txt
RUN pip install --no-cache-dir -r /code/requirements.txt

COPY ./src /code/src
COPY ./models /code/models
COPY ./data/cleaned_telco_data.joblib /code/data/cleaned_telco_data.joblib

EXPOSE 8000

CMD ["uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "8000"]
