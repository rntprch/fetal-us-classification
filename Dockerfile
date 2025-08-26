# Use the official NVIDIA CUDA image with Python 3.10
FROM nvidia/cuda:12.2.2-cudnn8-devel-ubuntu20.04

# Set the working directory in the container
WORKDIR /workspace

# Set DEBIAN_FRONTEND to noninteractive to avoid timezone prompts
ENV DEBIAN_FRONTEND=noninteractive

# Install required system packages and software-properties-common for add-apt-repository
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    wget \
    curl \
    ca-certificates \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    git \
    software-properties-common \
    && rm -rf /var/lib/apt/lists/*

# Add the deadsnakes PPA for Python 3.10
RUN add-apt-repository ppa:deadsnakes/ppa

# Install Python 3.10 and related packages
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3.10-dev \
    python3.10-distutils \
    && rm -rf /var/lib/apt/lists/*

# Install pip for Python 3.10
RUN curl https://bootstrap.pypa.io/get-pip.py | python3.10

# Update alternatives to use Python 3.10
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1 && \
    update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1

# Create symbolic link for pip
RUN ln -sf /usr/local/bin/pip3.10 /usr/bin/pip3 && \
    ln -sf /usr/local/bin/pip3.10 /usr/bin/pip

# Upgrade pip
RUN python3 -m pip install --upgrade pip

# Install packaging before all pips
RUN pip3 install packaging

# Install torch first
RUN pip3 install torch==1.13.0

# Copy the requirements.txt file into the container
COPY requirements.txt .

# Install the required Python packages
RUN pip3 install -r requirements.txt

# Install Jupyter Notebook
RUN pip3 install jupyter jupyterlab

# Install ipywidgets
RUN pip install --no-cache-dir ipywidgets

# Expose the port for Jupyter Notebook
EXPOSE 510

# Start Jupyter Notebook
CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--port=510", "--no-browser", "--allow-root"]