all: build run

build:
	nvcc -o main main.cu reduction.cu

run:
	./main

clean:
	rm -f main
