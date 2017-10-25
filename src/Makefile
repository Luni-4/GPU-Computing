MAIN = main
NVCC = nvcc
CUDAFLAGS = -std=c++11 -Xcompiler -Wall -arch=sm_20

OBJECTS := $(wildcard *.cu)

all: $(MAIN)

$(MAIN): $(OBJECTS)
	$(NVCC) $(CUDAFLAGS) $^ -o $@

.PHONY: clean

clean:
	rm -f *.o $(MAIN) 

