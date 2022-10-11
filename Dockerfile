FROM python:3.8-slim-buster

RUN apt-get update && apt-get install -y \
    build-essential 

RUN pip install --upgrade pip 
RUN pip install torch torchvision torchsummary numpy tqdm matplotlib 

WORKDIR /app
COPY *.py /app
COPY ./model /app/model

