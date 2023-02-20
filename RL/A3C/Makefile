#!/bin/bash

all: test

test:
	cd ./A3C-discrete && make test
	cd ./A3C-continuous && make test

clean:
	cd ./A3C-discrete && make clean
	cd ./A3C-continuous && make clean
	rm -rf ./__pycache__

