#!/bin/bash

all: test

test:
	cd ./CartPole && make test
	cd ./ContinuePendulum && make test

clean:
	cd ./CartPole && make clean
	cd ./ContinuePendulum && make clean
	rm -rf ./__pycache__

