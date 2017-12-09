MAIN = main
NVCC = nvcc
CUDAFLAGS = --default-stream per-thread -m 64 --fmad=false -std=c++11 -Xcompiler -Wall -O3 -arch=sm_20
LIBS = -lcublas

OBJECTS := $(wildcard src/*.cu)

ifeq ($(DEBUG), 1)
    CUDAFLAGS += -D DEBUG
endif

ifeq ($(TOYINPUT), 1)
    CUDAFLAGS += -D DEBUG,TOYINPUT
endif

all: $(MAIN)

$(MAIN): $(OBJECTS)
	$(NVCC) $(CUDAFLAGS) $^ -o $@ $(LIBS)

.PHONY: clean

clean:
	rm -f *.o $(MAIN) 

