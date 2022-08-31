#!/bin/bash

all: run

run:
	python generator.py
	python FastWeightsRNN.py

clean:
	rm -rf *.pyc
	rm -rf *.pkl
	rm -rf ./__pycache__

