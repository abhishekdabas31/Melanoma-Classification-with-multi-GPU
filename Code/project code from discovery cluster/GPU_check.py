## This notebook will display the details about the GPU

# imports

import torch

# GPU CHECK

# cuda version
print("cuda version: " , torch.version.cuda)

# check fot the GPU
print("GPU available: ",torch.cuda.is_available())

# check device name
print("Device Name : ", torch.cuda.get_device_name(0))
print("Device Name : ", torch.cuda.get_device_name(1))

# how many GPU's do we have
print("The number of GPU's we have are : ", torch.cuda.device_count() )

# switiching to GPU

if torch.cuda.is_available():
    device = torch.device("cuda")
    print("Running on the GPU")
else:
    device = torch.device("cpu")
    print("Running on the CPU")