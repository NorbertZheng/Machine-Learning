#!/usr/bin/env python3
"""
Created on 16:51, Nov. 10th, 2022

@author: Norbert Zheng
"""
import argparse

__all__ = [
    "args",
]

# def args macro
args = argparse.ArgumentParser()
args.add_argument("--dataset", default="cora")
args.add_argument("--model", default="gae")
args = args.parse_args()

if __name__ == "__main__":
    print(args)

