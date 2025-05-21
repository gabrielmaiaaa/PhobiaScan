FROM tensorflow/tensorflow:latest-gpu

RUN pip install --no-cache-dir --upgrade \
    "tensorflow[and-cuda]" \
    matplotlib \
    scipy \
    scikit-image \
    numpy \
    seaborn \
    scikit-learn \
    keras_tuner \
    pandas