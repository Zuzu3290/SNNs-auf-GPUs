FROM ubuntu:24.04

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.12 \
        python3.12-dev \
        python3-pip \
        build-essential \
        libhdf5-dev \
        git \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.12 /usr/bin/python

WORKDIR /app


RUN pip install --no-cache-dir --break-system-packages torch torchvision --index-url https://download.pytorch.org/whl/cu128

# tonic==1.6.0 (its own latest release) declares numpy<2.0.0, which conflicts with
# requirements.txt's numpy==2.3.5 if resolved together -- see that file's "Event data"
# section. --no-deps installs it without pip enforcing that stale constraint.
RUN pip install --no-cache-dir --break-system-packages --no-deps tonic==1.6.0

COPY requirements.txt .
RUN pip install --no-cache-dir --break-system-packages -r requirements.txt

COPY . .

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Docker's default 64MB /dev/shm caps DataLoader workers -- run with e.g. --shm-size=2g for real batch sizes.
CMD ["/bin/bash"]
