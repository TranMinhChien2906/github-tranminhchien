FROM nvidia/cuda:12.2.0-runtime-ubuntu20.04
ARG DEBIAN_FRONTEND=noninteractive
# install utilities
RUN apt-get update && \
    apt-get install --no-install-recommends -y curl

RUN apt-get update && apt-get install -y libstdc++6

ENV CONDA_AUTO_UPDATE_CONDA=false \
    PATH=/opt/miniconda/bin:$PATH

#https://repo.anaconda.com/miniconda/Miniconda3-py310_24.5.0-0-Linux-x86_64.sh new
#https://repo.anaconda.com/miniconda/Miniconda3-py38_4.9.2-Linux-x86_64.sh old
RUN curl -sLo ~/miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-py310_24.5.0-0-Linux-x86_64.sh \
    && chmod +x ~/miniconda.sh \
    && ~/miniconda.sh -b -p /opt/miniconda \
    && rm ~/miniconda.sh \
    && sed -i "$ a PATH=/opt/miniconda/bin:\$PATH" /etc/environment

# Installing python dependencies
RUN python3 -m pip --no-cache-dir install --upgrade pip && \
    python3 --version && \
    pip3 --version

# RUN pip3 --timeout=300 --no-cache-dir install torch==1.10.0+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html

COPY requirements.txt .
# RUN pip3 --timeout=300 --no-cache-dir install -r requirements.txt
# RUN pip install --no-cache-dir -r requirements.txt
RUN mkdir /code
COPY . /code/
COPY ./requirements.txt /code/
WORKDIR /code
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install python-multipart
RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y git  && \
    apt-get install -y tzdata &&\
    apt-get install -y libgl1-mesa-glx &&\
    apt-get install ffmpeg libsm6 libxext6  -y &&\
    apt-get install poppler-utils -y &&\
    apt-get install -y build-essential &&\
    apt-get install -y cmake &&\
    apt-get install -y python3-dev
# RUN pip install cmake
RUN pip install dlib
# RUN apt-get install -y6 libglib2.0-dev
# RUN apt-get update && apt-get install -y python3-opencv
RUN pip install opencv-python
# RUN install opencv-contrib-python
EXPOSE 9000
# CMD ["/start.sh"]
# docker build --compress -t ml/project1:latest 
