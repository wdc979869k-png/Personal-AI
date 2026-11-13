# Personal AI Project

## Project Introduction

This project is a personal AI chatbot trained
using my own personal information dataset.

The purpose is to create an AI model that can
answer questions about my interests, personality,
strengths, and future goals.

## Technologies

- Python
- PyTorch
- GRU Neural Network
- Streamlit
- Pandas

## Model Architecture

This project uses a GRU-based Encoder-Decoder
neural network.

The Encoder reads the input question.
The Decoder generates an answer character by character.

The model is trained from scratch using my own dataset.

## Dataset

The training dataset is stored in training_data.csv.

It contains questions, answers, and fact categories.

## Training

Run the following command:

python train.py

The trained model will be saved as:

model_weights.pth

## Run the Website

Install the required packages:

pip install -r requirements.txt

Start the Streamlit application:

streamlit run app.py

## Limitations

The model uses a small character-level dataset.
It may produce inaccurate or incomplete answers
when questions contain unfamiliar words.

More training examples and improved model
architecture may improve its performance.
