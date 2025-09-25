#!/bin/bash
# stops the script if an error occurs
set -e  

echo "Start training ..."
python train.py  --batch_size 8 --epochs 100

echo "Training completed."

echo "Start test ..."
python test.py  --split test

echo "test completed"

echo "Start validation ..."
python test.py  --split val
echo "validation completed <3 <3 <3"