# kaggleのpython環境をベースにする
FROM gcr.io/kaggle-images/python:v73

# 追加インストール
RUN pip install -U pip && \
    pip install fastprogress japanize-matplotlib invoke