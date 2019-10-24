FROM ubuntu:16.04
MAINTAINER tmacrina "tmacrina@zetta.ai"

ENV LC_CTYPE en_US.UTF-8
ENV LANG en_US.UTF-8

RUN apt-get update && apt-get install -y \
	language-pack-en \
	python3-pip \
	python3-dev \
	git \
	g++ \
  && cd /usr/local/bin \
  && ln -s /usr/bin/python3 python \
  && pip3 install --upgrade pip
WORKDIR /root
# Install igneous
RUN git clone https://github.com/seung-lab/igneous.git
WORKDIR /root/igneous
RUN pip --no-cache-dir install --upgrade \
	numpy \
	setuptools \
	wheel \
	cython
RUN pip install -r requirements.txt
RUN python setup.py develop 
# Install waterz
RUN apt-get install -y \
	libboost-dev \
	libboost-all-dev
WORKDIR /root
RUN git clone https://github.com/funkey/waterz.git
WORKDIR /root/waterz
RUN python setup.py develop
# Add segmentation script
WORKDIR /root
ADD . /root/seg
WORKDIR /root/seg
