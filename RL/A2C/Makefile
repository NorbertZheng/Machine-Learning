#!/bin/bash

all: test

test:
	cd ./A2C-discrete && make test
	cd ./A2C-continuous && make test

clean:
	cd ./A2C-discrete && make clean
	cd ./A2C-continuous && make clean
	rm -rf ./__pycache__

