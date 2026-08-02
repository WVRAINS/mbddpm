FROM pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime

WORKDIR /app

COPY . /app

RUN pip install --upgrade pip

RUN pip install . --no-deps

RUN pip install \
    numpy==1.26.4 \
    "pandas>=1.5" \
    tqdm==4.66.4 \
    pyyaml==6.0.1 \
    "timm>=0.9" \
    einops==0.8.2 \
    "openpyxl>=3.1"

CMD ["bash"]