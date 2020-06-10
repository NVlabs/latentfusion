FROM nvidia/cudagl:10.1-devel

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y locales
# Set the locale
RUN sed -i -e 's/# en_US.UTF-8 UTF-8/en_US.UTF-8 UTF-8/' /etc/locale.gen && \
    locale-gen
ENV LANG en_US.UTF-8
ENV LANGUAGE en_US:en
ENV LC_ALL en_US.UTF-8

RUN set -xe \
    && apt-get update -q \
    && apt-get upgrade -y \
    && apt-get install -y software-properties-common \
    && apt-get update -q

RUN set -xe \
    && apt-get remove -y python3 \
    && apt-get autoremove -y \
    && apt-get install --no-install-recommends -y \
            build-essential cmake libassimp-dev \
            libtiff5-dev libjpeg8-dev zlib1g-dev \
            libfreetype6-dev liblcms2-dev libwebp-dev libharfbuzz-dev libfribidi-dev \
            tcl8.6-dev tk8.6-de curl git freeglut3-dev

RUN set -xe \
    && apt-get install --no-install-recommends -y tmux vim

ENV PATH /opt/conda/bin:$PATH

RUN apt-get update --fix-missing && \
    apt-get install -y wget bzip2 ca-certificates curl git && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-py38_4.8.2-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh && \
    /opt/conda/bin/conda clean -tipsy && \
    ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
    echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc && \
    echo "conda activate base" >> ~/.bashrc

WORKDIR /app
COPY environment.yml environment.yml
RUN conda env create -f environment.yml -n latentfusion
RUN echo "source activate latentfusion" > ~/.bashrc
ENV PATH /opt/conda/envs/latentfusion/bin:$PATH

ENV LD_LIBRARY_PATH="/usr/lib/nvidia:/usr/lib/nvidia-418:${LD_LIBRARY_PATH}"
ENV PYTHONPATH /app/neurend:$PYTHONPATH

# Install PyOpenGL accelerate.
WORKDIR /app
RUN git clone https://github.com/keunhong/pyopengl.git
WORKDIR /app/pyopengl/accelerate
RUN python setup.py install
RUN python -c "from matplotlib import pyplot"

WORKDIR /app/neurend
COPY . /app/neurend
