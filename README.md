# State-of-the-art result for all Machine Learning Problems

### LAST UPDATE: 9th November, 2017

This repository provides state-of-the-art (SoTA) results for all machine learning problems. We do our best to keep this repository up to date.  If you do find a problem's SoTA result is out of date or missing, please raise this as an issue (with this information: research paper name, dataset, metric, source code and year). We will fix it immediately.

This is an attempt to make  one stop for all types of machine learning problems state of the art result.

This summary is categorized into:

- Supervised Learning
    - Speech
    - Computer Vision
    - NLP
- Unsupervised Learning
    - Speech
    - Computer Vision
    - NLP
- Transfer Learning
- Reinforcement Learning

## Supervised Learning


### NLP
#### 1. Language Modelling
Research Paper | Datasets  | Metric | Source Code | Year
------------ | ------------- | ------------ | -------------  | -------------
[Averaged Stochastic Gradient  Descent <br/> with Weight Dropped LSTM or QRNN](https://arxiv.org/pdf/1709.07432.pdf) | <ul><li> PTB </li><li> WikiText-2 </li></ul> | <ul><li> Preplexity: 51.1 </li><li> Perplexity: 44.3 </li></ul> |  [Pytorch](https://github.com/benkrause/dynamic-evaluation) | 2017
[Averaged Stochastic Gradient  Descent <br/> with Weight Dropped LSTM or QRNN](https://arxiv.org/pdf/1708.02182.pdf) | <ul><li> PTB </li><li> WikiText-2 </li></ul> | <ul><li> Preplexity: 52.8 </li><li> Perplexity: 52.0 </li></ul> |  [Pytorch](https://github.com/salesforce/awd-lstm-lm) | 2017
[FRATERNAL DROPOUT](https://arxiv.org/pdf/1711.00066.pdf) | <ul><li> PTB </li><li> WikiText-2 </li></ul> | <ul><li> Preplexity: 56.8 </li><li> Perplexity: 64.1</li></ul> |  NOT AVAILABLE YET | 2017
[Factorization tricks for LSTM networks](https://arxiv.org/pdf/1703.10722.pdf) |One Billion Word Benchmark |  Preplexity:  23.36 | [Tensorflow](https://github.com/okuchaiev/f-lm) | 2017




#### 2. Machine Translation
Research Paper | Datasets  | Metric | Source Code | Year
------------ | ------------- | ------------ | ------------- | -------------
[Attention Is All You Need](https://arxiv.org/abs/1706.03762) | <ul><li> WMT 2014 English-to-French </li><li> WMT 2014 English-to-German </li></ul> | <ul><li> BLEU: 41.0 </li><li> BLEU: 28.4</li></ul> |  <ul><li> [PyTorch](https://github.com/jadore801120/attention-is-all-you-need-pytorch) </li><li> [Tensorflow](https://github.com/tensorflow/tensor2tensor) </li></ul> | 2017

#### 3. Text Classification
Research Paper | Datasets  | Metric | Source Code | Year
------------ | ------------- | ------------ | ------------- | -------------
[Attentive Convolution](https://arxiv.org/pdf/1710.00519.pdf) | Yelp | Accuracy: 67.36 | NOT AVAILABLE YET | 2017


#### 4. Natural Language Inference
Research Paper | Datasets  | Metric | Source Code | Year
------------ | ------------- | ------------ | ------------- | -------------
[DiSAN: Directional Self-Attention Network <br /> for RNN/CNN-free Language Understanding](https://arxiv.org/pdf/1709.04696.pdf) | Stanford Natural Language Inference (SNLI) | Accuracy: 51.72 | NOT AVAILABLE YET | 2017

#### 5. Question Answering
Research Paper | Datasets  | Metric | Source Code | Year
------------ | ------------- | ------------ | ------------- | -------------
[Interactive AoA Reader+ (ensemble)](https://rajpurkar.github.io/SQuAD-explorer/) | The Stanford Question Answering Dataset | <ul><li> Exact Match: 79.083 </li><li> F1: 86.450 </li></ul>  | NOT AVAILABLE YET | 2017



#### 6. Named entity recognition

Research Paper | Datasets  | Metric | Source Code | Year
------------ | ------------- | ------------ | ------------- | -------------
[Named Entity Recognition in Twitter using Images and Text](https://arxiv.org/pdf/1710.11027.pdf) | Ritter | F-measure: 0.59 |[NLTK](https://github.com/aritter/twitter_nlp) | 2017




## Unsupervised Learning


## Transfer Learning


## Reinforcement Learning


Email: redditsota@gmail.com 
