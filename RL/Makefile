#!/bin/bash

all: test

test:
	cd ./ActorCritic && make test
	cd ./DeepQNetwork && make test
	cd ./PolicyGradient && make test
	cd ./QTableLearning && make test
	cd ./A2C && make test
	cd ./MetaRL && make test

clean:
	rm -rf ./__pycache__
	cd ./ActorCritic && make clean
	cd ./DeepQNetwork && make clean
	cd ./PolicyGradient && make clean
	cd ./QTableLearning && make clean
	cd ./A2C && make clean
	cd ./MetaRL && make clean
	rm -rf ./__pycache__

