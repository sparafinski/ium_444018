FROM ubuntu:latest

RUN apt-get update
RUN apt-get install -y python3-pip
RUN apt-get install -y unzip

RUN pip3 install kaggle
RUN pip3 install pandas
RUN pip3 install sklearn
RUN pip3 install numpy

RUN pip3 install matplotlib
RUN pip3 install torch
RUN pip3 install sacred
RUN pip3 install pymongo
RUN pip3 install mlflow
RUN pip3 install GitPython


ARG CUTOFF
ARG KAGGLE_USERNAME
ARG KAGGLE_KEY
ENV CUTOFF=${CUTOFF}
ENV KAGGLE_USERNAME=${KAGGLE_USERNAME}
ENV KAGGLE_KEY=${KAGGLE_KEY}

WORKDIR /app

COPY lab2/download.sh .
COPY lab6/biblioteka_DL/dllib.py .
COPY lab6/biblioteka_DL/evaluate.py .
COPY biblioteka_DL/imdb_top_1000.csv .
COPY predict.py .
COPY registry.py .

RUN chmod +x ./download.sh
RUN ./download.sh

RUN pip3 install dvc
RUN pip3 install dvc[ssh] paramiko
RUN apt install -y sshpass openssh-client
RUN useradd -r -u 111 jenkins

#CMD python3 ./dllib.py
