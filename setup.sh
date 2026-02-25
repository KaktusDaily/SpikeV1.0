#!/bin/bash

# Create virtual environment
python -m venv chatbot_env
source chatbot_env/bin/activate  # On Windows: chatbot_env\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the chatbot
python main.py
