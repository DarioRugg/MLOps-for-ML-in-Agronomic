FROM tensorflow/tensorflow:2.0.4-gpu-py3

RUN rm /etc/apt/sources.list.d/cuda.list
RUN rm /etc/apt/sources.list.d/nvidia-ml.list

RUN apt-get update && apt-get upgrade -y
RUN pip install --upgrade pip 

RUN apt-get install htop -y

RUN apt-get install git -y

RUN apt-get install tmux -y

WORKDIR /tmp/
COPY ./requirements_gpu.txt .

VOLUME ["/tokens"]

RUN pip install -r requirements_gpu.txt

RUN rm ./requirements_gpu.txt


ARG USERNAME=ruggeri
ARG USER_UID=2565
ARG USER_GID=$USER_UID

# Create the user
RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME \
    && apt-get install -y sudo \
    && echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME

# [Optional] Set the default user. Omit if you want to keep the default as root.
# USER $USERNAME