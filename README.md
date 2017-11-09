# State-of-the-art result for all Machine Learning Problems

### LAST UPDATE: 9th November, 2017

This repository provides state-of-the-art (SoTA) results for all machine learning problems. We do our best to keep this repository up to date.  If you do find a problem's SoTA result is out of date or missing, please raise this as an issue (with this information: research paper name, dataset, metric, source code and year). We will fix it immediately.

You can also submit this [Google Form](https://docs.google.com/forms/d/e/1FAIpQLSeMnkZ24YqiNkQEER_ihckenijBP7GpQpv8ZrkBnY7ythCItw/viewform?usp=sf_link) if you are new to Github.

This is an attempt to make  one stop for all types of machine learning problems state of the art result. I can not do this alone. I need help from everyone. Please submit the Google form/raise an issue if you find SOTA result for a dataset.  Please share this on Twitter, Facebook, and other social media.


This summary is categorized into:

- [Supervised Learning](https://github.com/RedditSota/state-of-the-art-result-for-machine-learning-problems#supervised-learning)
    - [Speech](https://github.com/RedditSota/state-of-the-art-result-for-machine-learning-problems#speech)
    - [Computer Vision](https://github.com/RedditSota/state-of-the-art-result-for-machine-learning-problems#computer-vision)
    - [NLP](https://github.com/RedditSota/state-of-the-art-result-for-machine-learning-problems#nlp)
- [Unsupervised Learning](https://github.com/RedditSota/state-of-the-art-result-for-machine-learning-problems#unsupervised-learning)
    - Speech
    - Computer Vision
    - NLP
- [Transfer Learning](https://github.com/RedditSota/state-of-the-art-result-for-machine-learning-problems#transfer-learning)
- [Reinforcement Learning](https://github.com/RedditSota/state-of-the-art-result-for-machine-learning-problems#reinforcement-learning)

## Supervised Learning


### NLP
#### 1. Language Modelling

<table>
  <tbody>
    <tr>
      <th width="30%">Research Paper</th>
      <th align="center" width="20%">Datasets</th>
      <th align="center" width="20%">Metric</th>
      <th align="center" width="20%">Source Code</th>
      <th align="center" width="10%">Year</th>
    </tr>
    <tr>
      <td><a href='https://arxiv.org/pdf/1709.07432.pdf'>DYNAMIC EVALUATION OF NEURAL SEQUENCE MODELS </a></td>
      <td align="left"><ul><li> PTB </li><li> WikiText-2 </li></ul></td>
      <td align="left"><ul><li> Preplexity: 51.1 </li><li> Perplexity: 44.3 </li></ul></td>
      <td align="left"><a href='https://github.com/benkrause/dynamic-evaluation'>Pytorch </a></td>
      <td align="left">2017</td>   
    </tr>
    <tr>
      <td><a href='https://arxiv.org/pdf/1708.02182.pdf'>Averaged Stochastic Gradient  Descent <br/> with Weight Dropped LSTM or QRNN </a></td>
      <td align="left"><ul><li> PTB </li><li> WikiText-2 </li></ul></td>
      <td align="left"><ul><li> Preplexity: 52.8 </li><li> Perplexity: 52.0 </li></ul></td>
      <td align="left"><a href='https://github.com/salesforce/awd-lstm-lm'>Pytorch </a></td>
      <td align="left">2017</td>   
    </tr>
    <tr>
      <td><a href='https://arxiv.org/pdf/1711.00066.pdf'>FRATERNAL DROPOUT </a></td>
      <td align="left"><ul><li> PTB </li><li> WikiText-2 </li></ul></td>
      <td align="left"><ul><li> Preplexity: 56.8 </li><li> Perplexity: 64.1 </li></ul></td>
      <td align="left"> <a href='https://github.com/kondiz/fraternal-dropout'> Pytorch </a>  </td>
      <td align="left">2017</td>   
    </tr>
        <tr>
      <td><a href='https://arxiv.org/pdf/1711.00066.pdf'>Factorization tricks for LSTM networks </a></td>
      <td align="left">One Billion Word Benchmark</td>
      <td align="left"> Preplexity:  23.36</td>
      <td align="left"><a href='https://github.com/okuchaiev/f-lm'>Tensorflow </a></td>
      <td align="left">2017</td>   
    </tr>
  </tbody>
</table>




#### 2. Machine Translation
Research Paper | Datasets  | Metric | Source Code | Year
------------ | ------------- | ------------ | ------------- | -------------
[Attention Is All You Need](https://arxiv.org/abs/1706.03762) | <ul><li> WMT 2014 English-to-French </li><li> WMT 2014 English-to-German </li></ul> | <ul><li> BLEU: 41.0 </li><li> BLEU: 28.4</li></ul> |  <ul><li> [PyTorch](https://github.com/jadore801120/attention-is-all-you-need-pytorch) </li><li> [Tensorflow](https://github.com/tensorflow/tensor2tensor) </li></ul> | 2017


#### 3. Text Classification

<table>
  <tbody>
    <tr>
      <th width="30%">Research Paper</th>
      <th align="center" width="20%">Datasets</th>
      <th align="center" width="20%">Metric</th>
      <th align="center" width="20%">Source Code</th>
      <th align="center" width="10%">Year</th>
    </tr>
    <tr>
      <td><a href='https://arxiv.org/abs/1705.09207'> Learning Structured Text Representations </a></td>
      <td align="left">Yelp</td>
      <td align="left">Accuracy: 68.6</td>
      <td align="left"> NOT YET AVAILABLE</td>
      <td align="left">2017</td>    
    </tr>
    <tr>
      <td><a href='https://arxiv.org/pdf/1710.00519.pdf'>Attentive Convolution</a></td>
      <td align="left">Yelp</td>
      <td align="left">Accuracy: 67.36</td>
      <td align="left"> NOT YET AVAILABLE</td>
      <td align="left">2017</td>   
    </tr>
  </tbody>
</table>

#### 4. Natural Language Inference
Research Paper | Datasets  | Metric | Source Code | Year
------------ | ------------- | ------------ | ------------- | -------------
[DiSAN: Directional Self-Attention Network <br /> for RNN/CNN-free Language Understanding](https://arxiv.org/pdf/1709.04696.pdf) | Stanford Natural Language Inference (SNLI) | Accuracy: 51.72 | NOT YET AVAILABLE | 2017

#### 5. Question Answering
Research Paper | Datasets  | Metric | Source Code | Year
------------ | ------------- | ------------ | ------------- | -------------
[Interactive AoA Reader+ (ensemble)](https://rajpurkar.github.io/SQuAD-explorer/) | The Stanford Question Answering Dataset | <ul><li> Exact Match: 79.083 </li><li> F1: 86.450 </li></ul>  | NOT YET AVAILABLE | 2017



#### 6. Named entity recognition

Research Paper | Datasets  | Metric | Source Code | Year
------------ | ------------- | ------------ | ------------- | -------------
[Named Entity Recognition in Twitter <br /> using Images and Text](https://arxiv.org/pdf/1710.11027.pdf) | Ritter | F-measure: 0.59 | NOT YET AVAILABLE | 2017


### Computer Vision

#### 1. Classification

Research Paper | Datasets  | Metric | Source Code | Year
------------ | ------------- | ------------ | ------------- | -------------
[Dynamic Routing Between Capsules](https://arxiv.org/pdf/1710.09829.pdf) | MNIST  | Test Error: 0.25±0.005 | <ul><li> [PyTorch](https://github.com/gram-ai/capsule-networks) </li><li> [Tensorflow](https://github.com/naturomics/CapsNet-Tensorflow) </li><li> [Keras](https://github.com/XifengGuo/CapsNet-Keras) </li><li>[Chainer](https://github.com/soskek/dynamic_routing_between_capsules) </li></ul>  | 2017
[High-Performance Neural Networks for Visual Object Classification](https://arxiv.org/pdf/1102.0183.pdf) | NORB  | Test Error: 2.53 ± 0.40| NOT FOUND | 2011
[Dynamic Routing Between Capsules](https://arxiv.org/pdf/1710.09829.pdf) | CIFAR-10  | Test Error: 10.6% | <ul><li> [PyTorch](https://github.com/gram-ai/capsule-networks) </li><li> [Tensorflow](https://github.com/naturomics/CapsNet-Tensorflow) </li><li> [Keras](https://github.com/XifengGuo/CapsNet-Keras) </li><li>[Chainer](https://github.com/soskek/dynamic_routing_between_capsules) </li></ul>  | 2017
[Dynamic Routing Between Capsules](https://arxiv.org/pdf/1710.09829.pdf) | MultiMNIST  | Test Error: 5% | <ul><li> [PyTorch](https://github.com/gram-ai/capsule-networks) </li><li> [Tensorflow](https://github.com/naturomics/CapsNet-Tensorflow) </li><li> [Keras](https://github.com/XifengGuo/CapsNet-Keras) </li><li>[Chainer](https://github.com/soskek/dynamic_routing_between_capsules) </li></ul>  | 2017
### Speech
#### 1. ASR
Research Paper | Datasets  | Metric | Source Code | Year
------------ | ------------- | ------------ | ------------- | -------------
[The Microsoft 2017 Conversational Speech Recognition System](https://arxiv.org/pdf/1708.06073.pdf) | Switchboard Hub5'00  | WER: 5.1 | NOT FOUND | 2017




## Unsupervised Learning

#### Computer Vision
##### 1. Generative Model
<table>
  <tbody>
    <tr>
      <th width="30%">Research Paper</th>
      <th align="center" width="20%">Datasets</th>
      <th align="center" width="20%">Metric</th>
      <th align="center" width="20%">Source Code</th>
      <th align="center" width="10%">Year</th>
    </tr>
    <tr>
      <td><a href='http://research.nvidia.com/sites/default/files/publications/karras2017gan-paper-v2.pdf'> PROGRESSIVE GROWING OF GANS FOR IMPROVED
QUALITY, STABILITY, AND VARIATION </a></td>
      <td align="left">Unsupervised CIFAR 10</td>
      <td align="left">Inception score: 8.80 </td>
      <td align="left"> <a='https://github.com/tkarras/progressive_growing_of_gans'>Theano</a></td>
      <td align="left">2017</td>    
    </tr>
  </tbody>
</table>

## Transfer Learning


## Reinforcement Learning


Email: redditsota@gmail.com 
