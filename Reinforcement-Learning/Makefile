#!/bin/bash

all: test

test:
	cd ./ActorCritic && make test
	cd ./DeepQNetwork && make test
	cd ./PolicyGradient && make test
	cd ./QTableLearning && make test

clean:
	cd ./ActorCritic && make clean
	cd ./DeepQNetwork && make clean
	cd ./PolicyGradient && make clean
	cd ./QTableLearning && make clean
	rm -rf ./__pycache__

