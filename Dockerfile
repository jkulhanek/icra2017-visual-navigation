FROM tensorflow/tensorflow:1.7.1

RUN pip install scikit-image
COPY . /app
WORKDIR /app