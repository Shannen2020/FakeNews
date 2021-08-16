## Overview
The Fake News Predictor aims to assist users in fact checking the credibility of news articles simply by supplying the URLs to these contents. Trained on labelled news articles in a supervised manner, we employ two models to help predict the likelihood of fake news using Neural Network Language Model (NNLM) and Gated Recurrent Units (GRU).
Users are offered the choice to either select the NNLM or GRU models to facilitate the assesment of news credibility.


## At a Glance
Fake news is everywhere online, here's offering a tool to help with your assessment!

## Likelihood of Fakes News
Predictions are tiered according to model confidence at 4 levels:
"Absolutely real."
"Might be real news."
"This might be fake news."
"Totally fake news!"

Non-English Articles
Current version does not support URLs with non-English contents. Users will be prompted to provide an English article alongside the detected language of the non-supported contents:
"That's <detected language>. English only!


## Data & Model Details
Data was obtained from Kaggle with deep learning architectures employed to build supervised-learners in fake news prediction.

## Data Sources
We use the Kaggle Fake News dataset, containing approximately 20,000 news articles which are labelled true and fake. A small number are non-English, which we take care for in both training and deployment.
Full credits to original contributor of dataset on Kaggle; as described by the original contributor:
train.csv: A full training dataset with the following attributes:

id:     unique id for a news article
title:  the title of a news article
author: author of the news article
text:   the text of the article; could be incomplete
label:  a label that marks the article as potentially unreliable

1: unreliable
0: reliable
Provided datasets were train.csv, test.csv and submit.csv. Since only train.csv contained labels, the models were trained on this dataset.

## Supervised Learning Models

Two models have been constructed, each with decent level of performances. Apart from architectural differences, the data processing pipelines adopted differ.

## Neural Network Language Model

The (probabilistic) neural network language model is a classic language model by Yoshua Bengio and others. In short, the NNLM is a neural architecture that simultaneously learns word embeddings and a statistical language model.
In our implementation we train the NNLM model using a token-based text embedding trained on Google News' 200 billion article corpus, freely available in Tensorflow Hub.
The training is rather fast, taking no more than 15 minutes for the dataset we use (see below). The downside is that the model payload is very large, at about 1 GB.

## Gated Recurrent Units

A simple RNN model with a embedding layer of 128 units and GRU of 128 units with softmax output for classification was used to train on test dataset. This yield a model size of ard 2mb compared to the NNLM above
Data & Model Pipeline
The train and test dataset was processed using keras' built-in preprocessing text tokenizer.
A max vocab dictionary of 1000 and a max sequence length of 500 is used to speed up the training pipeline
Training Time

Accuracy score on the test dataset is 0.9132905


## Contributors
Beng Hau, Wykeith, Shannen & Rui Ming, Team 2, AIAP Batch 8, AI Singapore
