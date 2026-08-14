FROM ubuntu:24.04

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.12 \
        python3-pip \
        build-essential \
        libhdf5-dev \
        git \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.12 /usr/bin/python

WORKDIR /app


RUN pip install --no-cache-dir --break-system-packages torch torchvision --index-url https://download.pytorch.org/whl/cu128

COPY requirements.txt .
RUN pip install --no-cache-dir --break-system-packages -r requirements.txt

COPY . .

ENV TF_CPP_MIN_LOG_LEVEL=3 \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

CMD ["/bin/bash"]
