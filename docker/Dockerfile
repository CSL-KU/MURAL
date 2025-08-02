FROM dustynv/l4t-pytorch:r36.3.0-cu124

ARG CUDA_ARCH

WORKDIR /root

RUN apt-get update
RUN apt-get install -y vim tmux python-is-python3 htop ninja-build tree python3-pip libopenblas-dev git cmake libgl1
RUN python3 -m pip install --index-url https://pypi.org/simple/ --upgrade pip
RUN pip install --no-cache -v --index-url https://pypi.org/simple/ pccm wheel SharedArray scikit-learn scikit-image av2==0.2.1 kornia==0.6.12 easydict tensorboardX alive-progress onnx==1.16.1 onnxruntime==1.20 numpy==1.26.4 motmetrics==1.1.3 pyquaternion

RUN wget https://developer.nvidia.com/downloads/compute/machine-learning/tensorrt/10.1.0/tars/TensorRT-10.1.0.27.l4t.aarch64-gnu.cuda-12.4.tar.gz
RUN tar xvf TensorRT-10.1.0.27.l4t.aarch64-gnu.cuda-12.4.tar.gz
RUN rm TensorRT-10.1.0.27.l4t.aarch64-gnu.cuda-12.4.tar.gz
RUN mv TensorRT-10.1.0.27 TensorRT
RUN echo "export LD_LIBRARY_PATH=/root/TensorRT/lib:$LD_LIBRARY_PATH" >> ~/.bashrc && \
    echo "export PATH=/root/TensorRT/bin:$PATH" >> ~/.bashrc
RUN python3 -m pip install TensorRT/python/tensorrt-10.1.0-cp310-none-linux_aarch64.whl

RUN git clone https://github.com/CSL-KU/MURAL
RUN git clone -b v0.4.11 https://github.com/FindDefinition/cumm.git
RUN git clone -b v2.3.6 https://github.com/traveller59/spconv.git
RUN git clone -b 2.1.2 https://github.com/rusty1s/pytorch_scatter.git
RUN git clone -b anytime_lidar https://github.com/ahmedius2/nuscenes-devkit.git
RUN git clone https://github.com/ahmedius2/sbnet.git

WORKDIR /root/nuscenes-devkit
RUN pip install --index-url https://pypi.org/simple/ -r setup/requirements.txt

WORKDIR /root/cumm
RUN git apply ../MURAL/req_patches/cumm.patch
RUN MAX_JOBS=$(nproc) CUMM_CUDA_ARCH_LIST=$CUDA_ARCH CUMM_DISABLE_JIT="1" python setup.py bdist_wheel
RUN pip install dist/*

WORKDIR /root/spconv
RUN MAX_JOBS=$(nproc) CUMM_CUDA_ARCH_LIST=$CUDA_ARCH SPCONV_DISABLE_JIT="1" python setup.py bdist_wheel
RUN pip install dist/*

WORKDIR /root/pytorch_scatter
RUN CPATH=$CPATH:/usr/local/cuda/include TORCH_CUDA_ARCH_LIST=$CUDA_ARCH FORCE_CUDA=1 pip install -v .

WORKDIR /root/sbnet/sbnet_pytorch
RUN MAX_JOBS=$(nproc) TORCH_CUDA_ARCH_LIST=$CUDA_ARCH pip install -v .

RUN echo "export PCDET_PATH=/root/MURAL" >> ~/.bashrc
WORKDIR /root/MURAL
RUN MAX_JOBS=$(nproc) TORCH_CUDA_ARCH_LIST=$CUDA_ARCH python setup.py develop

WORKDIR /root
RUN echo "export PYTHONPATH=\${PYTHONPATH}:\$HOME/nuscenes-devkit/python-sdk" >> ~/.bashrc
RUN echo "export PATH=\${PATH}:/usr/src/tensorrt/bin" >> ~/.bashrc

RUN sed -i 's/from collections import OrderedDict, Iterable/from collections import OrderedDict\nfrom collections.abc import Iterable/' \
	'/usr/local/lib/python3.10/dist-packages/motmetrics/metrics.py'

RUN echo 'if [ -e "/var/lib/nvpmodel/status" ]; then' >> ~/.bashrc && \
    echo '    export PMODE=$(cat "/var/lib/nvpmodel/status" | tr '"'"':'"'"' '"'"'_'"'"')' >> ~/.bashrc && \
    echo 'else' >> ~/.bashrc && \
    echo '    export PMODE="UNKNOWN_POWER_MODE"' >> ~/.bashrc && \
    echo 'fi' >> ~/.bashrc
