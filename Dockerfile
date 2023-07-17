# Base image with CUDA
FROM tensorflow/tensorflow:latest-gpu

# Set the working directory
WORKDIR /app
# Copy your Python code to the working directory
COPY . /app

RUN apt-get update -y && apt-get install -y \
    jq \
    python \
    pip

RUN pip3 install --upgrade pip
RUN pip3 install .

# Run the standardml module
CMD ["sh", "entrypoint.sh"]
