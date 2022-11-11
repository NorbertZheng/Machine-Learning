#!/bin/bash

all: test

test:
	cd ./GCN && make test
	cd ./GAE && make test

clean:
	rm -rf ./__pycache__
	cd ./GCN && make clean
	cd ./GAE && make clean

