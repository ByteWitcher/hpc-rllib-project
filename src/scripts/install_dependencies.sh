#!/bin/bash

# Location of virtual environment
VENV_DIR="$HOME/rllib_lab/rllib_venv"

if [ ! -d "$VENV_DIR" ]; then
    python3 -m venv "$VENV_DIR"
fi

source "$VENV_DIR/bin/activate"

pip install --upgrade pip

# Install dependencies
pip install --upgrade "numpy<2" pandas pydantic
pip install "ray[rllib]" torch

echo "Dependencies installed in $VENV_DIR"
