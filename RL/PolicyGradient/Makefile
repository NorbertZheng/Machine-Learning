#!/bin/bash

all: test

test:
	cd ./CartPole && make test
	cd ./MountainCar && make test
	python RL_brain.py

clean:
	cd ./CartPole && make clean
	cd ./MountainCar && make clean
	rm -rf ./__pycache__

