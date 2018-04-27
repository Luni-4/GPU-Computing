nvcc -m 64 -Xptxas -O3 -Xcompiler -Wall -Xcompiler /std:c++14 -Xcompiler /O2 -arch=sm_20 src\Cifar.cu src\Mnist.cu src\Network.cu src\ToyInput.cu src\FullyConnected.cu src\Convolutional.cu src\Kernel.cu src\KernelCPU.cu src\main.cu -o main560 -lcublas

PAUSE