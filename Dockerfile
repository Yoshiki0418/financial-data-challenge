# ベースイメージ
FROM ubuntu:22.04

# 必要なパッケージをインストール
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    git \
    curl \
    build-essential \
    && apt-get clean \
    apt-get install poppler-utils

# ワークディレクトリを設定
WORKDIR /workspace

# 必要なPythonライブラリをインストール（任意）
COPY requirements.txt /workspace
RUN pip install --no-cache-dir -r requirements.txt
