nvcc --default-stream per-thread -Xcompiler -Wall -Xcompiler /std:c++14 -Xcompiler /O2 -arch=sm_50 src\Cifar.cu src\Mnist.cu src\Network.cu src\ToyInput.cu src\FullyConnected.cu src\Convolutional.cu src\Kernel.cu src\KernelCPU.cu src\main.cu -o main860 -lcublas

PAUSE