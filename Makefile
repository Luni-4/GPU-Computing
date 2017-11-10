MAIN = main
NVCC = nvcc
CUDAFLAGS = -std=c++11 -Xcompiler -Wall -arch=sm_20
LIBS = -lcublas

OBJECTS := $(wildcard src/*.cu)

ifeq ($(DEBUG), 1)
    CUDAFLAGS += -DDEBUG
endif

all: $(MAIN)

$(MAIN): $(OBJECTS)
	$(NVCC) $(CUDAFLAGS) $^ -o $@ $(LIBS)

.PHONY: clean

clean:
	rm -f *.o $(MAIN) 

