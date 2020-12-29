# Computer Vision Classification Problem with GPU

## Files:
- ``Code folder`` - The complete code for the project
- ``Images``- Screenshots of the process
- ``Final Project Presentation`` - PPT
- ``Final report`` - Report 


## NVIDIA GPU
``nvidia-smi`` - check the details about the current nvidia gpu

## Running the files:
- Store the images in the scratch folder in you discovery or change the location of the image folder in the script
- running commands:
- ``srun -p reservation --reservation=csye7105-gpu --gres=gpu:4  --mem=120G  --pty /bin/bash``
- ``conda activate pytorch(name_of_your_env)``
- ``python model.py``


# Selecting the best GPU
- Memory bandwidth is the most important characteristic of a GPU. Opt for a GPU with the highest bandwidth available within your budget.
- The number of cores determines the speed at which the GPU can process data, the higher the number of cores, the faster the GPU can compute data. Consider this especially when dealing with large amounts of data.

## Compare the Parallel performance
- Elapsed time of same code segment
- Epochs

## Reading file:
- General Parallel File System (GPFS):
Parelle file systems historically have targeted high performance computing (HPC) environments that require access to large files, massive quantities from multiple computing servers. In Discovery it is the ``scratch directory``

## Computing 
- using srun
``srun -p --pty /bin/bash``
- sbatch
``sbatch batch_file``

## Worl Load Manager:
- SLURM - Simple Linux Utility Resource Management

## Reservation

### SRUN
``srun -p gpu --gres=gpu:1 --pty /bin/bash``
``srun -p reservation --reservation=csye7105-gpu --gres=gpu:1  --mem=100G  --pty /bin/bash``
``srun -p reservation --reservation=csye7105-gpu --gres=gpu:4  --mem=120G  --pty /bin/bash``
``srun --reservation=csye7105-gpu --gres=gpu:1 --pty /bin/bash``
``srun --partition=express  --nodes 1 --ntasks 10 --cpus-per-task 2 --pty --export=ALL --mem=1G --time=01:00:00 /bin/bash``

### SBATCH
``sbatch <scriptname.script>``
``scancel <jobid>``
``squeue -u <your user name>``

## Miniconda environment
``conda create -n myenv python=3.7``;
``Conda Activate myenv``

## Managing Anaconda Environment:
[anaconda](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html)

## transfering the local cuda env to jupyter notebook
``python3 -m ipykernel install --user --name=name_of_the_env``

## transfering the local cuda env to jupyter notebook
``python3 -m ipykernel``

## Installs with their commands 
- Torch environment:
``conda install pytorch torchvision torchaudio cudatoolkit=11.0 -c pytorch``
``conda install torchvision``
- CV2 Library 
``conda install -c conda-forge opencv``
- Albumentation library 
``conda install -c conda-forge albumentations``
- Pretrained Models Library
``conda install -c conda-forge pretrainedmodels``


## Other Important topics:

## Why Parallel Computing ?
Parallel computing is much suited for modelling and simulations of complex problems. Increse the processing power, to millions of transactions every second.

# Graphics Processing Unit(GPU)
A typical workload required arithmetic operations onlarge amount of data for rendering ans shading. It is designed to be goo with matrix operations. GPUs are optimized for training artificial intelligence and deep learning models as they can process multiple computations simultaneously.

## GPU vs CPU
1. A large part of CPU is dedicated to ``Control`` and ``Cache`` and smaller part is dedicated to Arithmetic Logic Unit(ALU)
1. GPU works best when the computation task can be split into parallel tasks. 
1. When the computation task is already small, we wont benifit from breaking the task and running on a GPU.
1. If your neural network is relatively small-scale, you can make do without a GPU
1. GPUs are a safer bet for fast machine learning because, at its heart, data science model training consists of simple matrix math calculations, the speed of which may be greatly enhanced if the computations are carried out in parallel.

[Myth Buster - CPU VS GPU ](https://www.youtube.com/watch?time_continue=93&v=-P28LKWTzrI&feature=emb_logo&ab_channel=NVIDIA)

### PyTorch
Pytorch is a deep learning framework and a scientific computing package. It is an open-source deep learning library. ``torch`` is the main module that holds all the things you need for Tensor computation.

- [Pytorch Better for research than Tensorflow](https://thegradient.pub/state-of-ml-frameworks-2019-pytorch-dominates-research-tensorflow-dominates-industry/)
- PyTorch Cuda
[Documentation](https://pytorch.org/docs/stable/cuda.html)

### CUDA
NVIDIA, the leader in manufacturing graphic cards, has created CUDA, which is a parallel computing Platform and Programming model. NVIDIA GPU's have CUDA extension which allows GPU support for Tensorflow and PyTorch.
- Different approaches to CUDA in python
1. Drop-in replacement
ex. CuPy
2. Compiling CUDA strings in python
3. C/C++ extension

### CUDA Semantics for PyTorch
``torch.cuda`` is used to set up and run CUDA operations. It keeps track of the currently selected GPU, and all CUDA tensors you allocate will by default be created on that device. The selected device can be changed with a ``torch.cuda.device`` context manager.
- Get the memory details of the GPU:
``torch.cuda.get_device_properties(device).total_memory``

- Example:
``cuda = torch.device("cuda:0")
  a1 = torch.tensor([1,2], device = cuda)
  a2 = torch.randn(2, device=cuda)``

``b1 = torch.tensor([1,2]).to(device= cuda)
  b2 = torch.randn(2, device=cuda) ``

## Numpy vs Tensors
1. Numpy Array
It is a package in python, which provides utilities for multidimensional arrays and matrices. It is n-dimensional array or ndarray. We use these array when we need to perform mathematical operations on all the elements.
1. Tensors
A scalar, vector, matrix, all are a tensor. They provide mathematical framework to solve problems in physics. A tensor is a multidimensional array with a uniform data type as dtype.

### USE:
- A Tensor is a suitable choice if you are going to use GPU. A tensor can reside in accelerator’s memory.
- Tensors are immutable. You always create new one.
- A vector is a one-dimensional or first order tensor and a matrix is a two-dimensional or second order tensor.
- Tensors contain data of ``uniform data type``
- Tensor conputations between tensors depend on the ``dtype`` and ``device``
- ``torch.tensor() ; torch.Tensor() ; torch.as_tensor() ; torch.from_numpy()``

### Vanishing Gradient:
- Since 2013, the Deep Learning community started to build deeper networks because they were able to achieve high accuracy values. Furthermore, deeper networks can represent more complex features, therefore the model robustness and performance can be increased. However, stacking up more layers didn’t work for the researchers. While training deeper networks, the problem of accuracy degradation was observed. In other words, adding more layers to the network either made the accuracy value to saturate or it abruptly started to decrease. The culprit for accuracy degradation was vanishing gradient effect which can only be observed in deeper networks.
- During the backpropagation stage, the error is calculated and gradient values are determined. The gradients are sent back to hidden layers and the weights are updated accordingly. The process of gradient determination and sending it back to the next hidden layer is continued until the input layer is reached. The gradient becomes smaller and smaller as it reaches the bottom of the network. Therefore, the weights of the initial layers will either update very slowly or remains the same. In other words, the initial layers of the network won’t learn effectively. Hence, deep network training will not converge and accuracy will either starts to degrade or saturate at a particular value. Although vanishing gradient problem was addressed using the normalized initialization of weights, deeper network accuracy was still not increasing.

