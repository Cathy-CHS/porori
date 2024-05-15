FROM pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime

# python 3.11
RUN apt-get update && apt-get install -y python3.11 python3.11-distutils

WORKDIR /porori

COPY ./ /porori/

RUN bash ./install.sh