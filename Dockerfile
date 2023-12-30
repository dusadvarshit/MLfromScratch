FROM python:3.10-slim-bullseye
COPY . /app  
WORKDIR /app
RUN pip install -r requirements.txt
CMD streamlit run main.py

