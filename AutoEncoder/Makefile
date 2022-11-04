#!/bin/bash

all: test

test:
	cd ./AE && make test
	cd ./VAE && make test

clean:
	rm -rf ./__pycache__
	cd ./AE && make clean
	cd ./VAE && make clean

