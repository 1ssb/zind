#!/bin/bash

set -e

SERVER_TOKEN="b9da59a1f76e5a7154f7656164d18579"
OUTPUT_FOLDER="/home/group/cvml-datasets/zind/data/"

python download_data.py -s "$SERVER_TOKEN" -o "$OUTPUT_FOLDER"

# TODO: Pipe the output to a log file for monitoring
