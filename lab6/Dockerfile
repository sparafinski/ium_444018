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

ARG CUTOFF
ARG KAGGLE_USERNAME
ARG KAGGLE_KEY
ENV CUTOFF=${CUTOFF}
ENV KAGGLE_USERNAME=${KAGGLE_USERNAME}
ENV KAGGLE_KEY=${KAGGLE_KEY}

WORKDIR /app

COPY lab2/download.sh .
COPY biblioteka_DL/dllib.py .
COPY biblioteka_DL/evaluate.py .
COPY biblioteka_DL/imdb_top_1000.csv .

RUN chmod +x ./download.sh
RUN ./download.sh
#CMD python3 ./dllib.py
