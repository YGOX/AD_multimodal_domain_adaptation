#!/bin/bash

python getpetCurrentLabel.py
python createpet3Dh5pyData.py
python getRIDtoPETmapping.py
python splitDataset.py

