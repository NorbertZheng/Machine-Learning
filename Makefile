#!/bin/bash

all: test

test:
	cd ./RL && make test
	cd ./Transformer && make test
	cd ./GNN && make test
	cd ./AutoEncoder && make test

clean:
	rm -rf ./__pycache__
	cd ./RL && make clean
	cd ./Transformer && make clean
	cd ./GNN && make clean
	cd ./AutoEncoder && make clean

