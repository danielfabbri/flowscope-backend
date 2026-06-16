#!/bin/bash

# Backend Run Script
# Simple startup without dev features

cd "$(dirname "$0")"

source venv/bin/activate
python main.py
