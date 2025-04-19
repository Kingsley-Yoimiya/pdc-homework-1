all: build run

build:
	nvcc -arch=compute_80 -code=sm_80 -o main.exe main.cu reduction.cu

run:
	./main.exe

clean:
	rm -f main.exe
