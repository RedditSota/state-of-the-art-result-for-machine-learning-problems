# State-of-the-art result for all Machine Learning Problems

### LAST UPDATE: 20th Februray 2019

### NEWS: I am looking for a Collaborator esp who does research in NLP, Computer Vision and Reinforcement learning. If you are not a researcher, but you are willing, contact me. Email me: yxt.stoaml@gmail.com

This repository provides state-of-the-art (SoTA) results for all machine learning problems. We do our best to keep this repository up to date.  If you do find a problem's SoTA result is out of date or missing, please raise this as an issue (with this information: research paper name, dataset, metric, source code and year). We will fix it immediately.

You can also submit this [Google Form](https://docs.google.com/forms/d/e/1FAIpQLSe_fFZVCeCVRGGgOQIpoQSXY7mZWynsx7g6WxZEVpO5vJioUA/viewform?embedded=true) if you are new to Github.

This is an attempt to make  one stop for all types of machine learning problems state of the art result. I can not do this alone. I need help from everyone. Please submit the Google form/raise an issue if you find SOTA result for a dataset.  Please share this on Twitter, Facebook, and other social media.


This summary is categorized into:

- [Supervised Learning](https://github.com/RedditSota/state-of-the-art-result-for-machine-learning-problems#supervised-learning)
    - [Speech](https://github.com/RedditSota/state-of-the-art-result-for-machine-learning-problems#speech)
    - [Computer Vision](https://github.com/RedditSota/state-of-the-art-result-for-machine-learning-problems#computer-vision)
    - [NLP](https://github.com/RedditSota/state-of-the-art-result-for-machine-learning-problems#nlp)
- [Semi-supervised Learning](https://github.com/RedditSota/state-of-the-art-result-for-machine-learning-problems#semi-supervised-learning)
    - Computer Vision
- [Unsupervised Learning](https://github.com/RedditSota/state-of-the-art-result-for-machine-learning-problems#unsupervised-learning)
    - Speech
    - Computer Vision
    - [NLP](https://github.com/RedditSota/state-of-the-art-result-for-machine-learning-problems/blob/master/README.md#nlp-1)
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
      <td><a href='https://d4mucfpksywv.cloudfront.net/better-language-models/language-models.pdf'> Language Models are Unsupervised Multitask Learners </a></td>
      <td align="left"><ul><li> PTB </li><li> WikiText-2 </li></ul></td>
      <td align="left"><ul><li> Perplexity: 35.76 </li><li> Perplexity: 18.34 </li></ul></td>
      <td align="left"><a href='https://github.com/openai/gpt-2'>Tensorflow </a></td>
      <td align="left">2019</td>   
    </tr>
    <tr>
      <td><a href='https://arxiv.org/pdf/1711.03953.pdf'>BREAKING THE SOFTMAX BOTTLENECK: A HIGH-RANK RNN LANGUAGE MODEL </a></td>
      <td align="left"><ul><li> PTB </li><li> WikiText-2 </li></ul></td>
      <td align="left"><ul><li> Perplexity: 47.69 </li><li> Perplexity: 40.68 </li></ul></td>
      <td align="left"><a href='https://github.com/zihangdai/mos'>Pytorch </a></td>
      <td align="left">2017</td>   
    </tr>
    <tr>
      <td><a href='https://arxiv.org/pdf/1709.07432.pdf'>DYNAMIC EVALUATION OF NEURAL SEQUENCE MODELS </a></td>
      <td align="left"><ul><li> PTB </li><li> WikiText-2 </li></ul></td>
      <td align="left"><ul><li> Perplexity: 51.1 </li><li> Perplexity: 44.3 </li></ul></td>
      <td align="left"><a href='https://github.com/benkrause/dynamic-evaluation'>Pytorch </a></td>
      <td align="left">2017</td>   
    </tr>
    <tr>
      <td><a href='https://arxiv.org/pdf/1708.02182.pdf'>Averaged Stochastic Gradient  Descent <br/> with Weight Dropped LSTM or QRNN </a></td>
      <td align="left"><ul><li> PTB </li><li> WikiText-2 </li></ul></td>
      <td align="left"><ul><li> Perplexity: 52.8 </li><li> Perplexity: 52.0 </li></ul></td>
      <td align="left"><a href='https://github.com/salesforce/awd-lstm-lm'>Pytorch </a></td>
      <td align="left">2017</td>   
    </tr>
    <tr>
      <td><a href='https://arxiv.org/pdf/1711.00066.pdf'>FRATERNAL DROPOUT </a></td>
      <td align="left"><ul><li> PTB </li><li> WikiText-2 </li></ul></td>
      <td align="left"><ul><li> Perplexity: 56.8 </li><li> Perplexity: 64.1 </li></ul></td>
      <td align="left"> <a href='https://github.com/kondiz/fraternal-dropout'> Pytorch </a>  </td>
      <td align="left">2017</td>   
    </tr>
        <tr>
      <td><a href='https://arxiv.org/pdf/1703.10722.pdf'>Factorization tricks for LSTM networks </a></td>
      <td align="left">One Billion Word Benchmark</td>
      <td align="left"> Perplexity:  23.36</td>
      <td align="left"><a href='https://github.com/okuchaiev/f-lm'>Tensorflow </a></td>
      <td align="left">2017</td>   
    </tr>
  </tbody>
</table>




#### 2. Machine Translation

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
      <td><a href='https://arxiv.org/pdf/1808.09381v2.pdf'> Understanding Back-Translation at Scale </a></td>
      <td align="left"> <ul><li>WMT 2014 English-to-French </li><li>WMT 2014 English-to-German </li></ul></td>
      <td align="left"> <ul><li>  BLEU: 45.6 </li><li>   BLEU: 35.0 </li></ul> </td>
      <td align="left"> <ul><li><a href='https://github.com/pytorch/fairseq'>PyTorch</a></li></ul></td>
      <td align="left">2018</td>    
    </tr>
    <tr>
      <td><a href='https://arxiv.org/pdf/1711.02132.pdf'>WEIGHTED TRANSFORMER NETWORK FOR
MACHINE TRANSLATION</a></td>
      <td align="left"> <ul><li>WMT 2014 English-to-French </li><li>WMT 2014 English-to-German </li></ul></td>
      <td align="left"> <ul><li>  BLEU: 41.4 </li><li>   BLEU: 28.9 </li></ul> </td>
      <td align="left"> <ul><li><a href=''>NOT FOUND</a></li></ul></td>
      <td align="left">2017</td>    
    </tr>
    <tr>
      <td><a href='https://arxiv.org/abs/1706.03762'>Attention Is All You Need</a></td>
      <td align="left"> <ul><li>WMT 2014 English-to-French </li><li>WMT 2014 English-to-German </li></ul></td>
      <td align="left"> <ul><li>  BLEU: 41.0 </li><li>   BLEU: 28.4 </li></ul> </td>
      <td align="left"> <ul><li><a href='https://github.com/jadore801120/attention-is-all-you-need-pytorch'>PyTorch</a> </li><li> <a href='https://github.com/tensorflow/tensor2tensor'>Tensorflow</a></li></ul></td>
      <td align="left">2017</td>    
    </tr>
     <tr>
      <td><a href='https://einstein.ai/static/images/pages/research/non-autoregressive-neural-mt.pdf'>NON-AUTOREGRESSIVE
NEURAL MACHINE TRANSLATION</a></td>
      <td align="left"> <ul><li> WMT16 Ro→En </li></ul></td>
      <td align="left"> <ul><li> BLEU: 31.44 </li></ul> </td>
      <td align="left"><ul><li><a href='https://github.com/salesforce/nonauto-nmt'>PyTorch</a></ul></li></td>
      <td align="left">2017</td>    
      </tr>
          <tr>
      <td><a href='https://arxiv.org/abs/1703.04887'> Improving Neural Machine Translation with Conditional Sequence Generative Adversarial Nets</a></td>
      <td align="left"> <ul><li>NIST02    </li><li>NIST03 </li><li>NIST04 </li><li>NIST05 </li></ul></td>
      <td align="left"><li>38.74  </li><li>36.01  </li><li> 37.54 </li><li>33.76 </li></ul </td>
      <td align="left"> <ul><li><a href='https://github.com/ngohoanhkhoa/GAN-NMT'>NMTPY</a> </li></ul></td>
      <td align="left">2017</td>    
    </tr>
  </tbody>
</table>  

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
      <td align="left"> <ul><li><a href='https://github.com/nlpyang/structured'>Tensorflow</a></ul></li></td>
      <td align="left">2017</td>    
    </tr>
    <tr>
      <td><a href='https://arxiv.org/pdf/1710.00519.pdf'>Attentive Convolution</a></td>
      <td align="left">Yelp</td>
      <td align="left">Accuracy: 67.36</td>
      <td align="left"> <ul><li><a href='https://github.com/yinwenpeng/Attentive_Convolution'>Theano</a></ul></li></td>
      <td align="left">2017</td>   
    </tr>
  </tbody>
</table>

#### 4. Natural Language Inference 
Leader board: 

[Stanford Natural Language Inference (SNLI)](https://nlp.stanford.edu/projects/snli/)

[MultiNLI](https://www.kaggle.com/c/multinli-matched-open-evaluation/leaderboard)

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
      <td><a href='https://arxiv.org/pdf/1709.04348.pdf'> NATURAL LANGUAGE INFERENCE OVER INTERACTION SPACE </a></td>
      <td align="left">Stanford Natural Language Inference (SNLI)</td>
      <td align="left">Accuracy: 88.9</td>
      <td align="left"><a href='https://github.com/YichenGong/Densely-Interactive-Inference-Network'>Tensorflow</a> </td>
      <td align="left">2017</td>
  </tr>
    <tr>
      <td><a href=https://arxiv.org/pdf/1810.04805.pdf> BERT-LARGE (ensemble) </a></td>
      <td align="left">Multi-Genre Natural Language Inference (MNLI)</td>
      <td align="left"><ul><li>Matched accuracy: 86.7</li><li>Mismatched accuracy: 85.9</td>
      <td align="left"><ul><li><a href='https://github.com/google-research/bert'>Tensorflow</a></li><li><a href='https://github.com/huggingface/pytorch-pretrained-BERT'>PyTorch</a></li> </td>
      <td align="left">2018</td>
  </tr>
  </tbody>
</table>


#### 5. Question Answering
Leader Board

[SQuAD](https://rajpurkar.github.io/SQuAD-explorer/)
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
      <td><a href='https://arxiv.org/pdf/1810.04805.pdf'> BERT-LARGE (ensemble) </a></td>
      <td align="left">The Stanford Question Answering Dataset</td>
      <td align="left"><ul><li> Exact Match: 87.4 </li><li> F1: 93.2 </li></ul></td>
      <td align="left"><ul><li><a href='https://github.com/google-research/bert'>Tensorflow</a></li><li><a href='https://github.com/huggingface/pytorch-pretrained-BERT'>PyTorch</a> </td>
      <td align="left">2018</td>    
  </tr>
  </tbody>
</table>


#### 6. Named entity recognition
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
      <td><a href='https://arxiv.org/pdf/1710.11027.pdf'>Named Entity Recognition in Twitter using Images and Text </a></td>
      <td align="left">Ritter</td>
      <td align="left"><ul><li> F-measure: 0.59 </li></ul></td>
      <td align="left"><a href=''>NOT FOUND</a> </td>
      <td align="left">2017</td>    
  </tr>
  </tbody>
</table>

#### 7. Abstractive Summarization

Research Paper | Datasets  | Metric | Source Code | Year  
------------ | ------------- | ------------ | ------------- | -------------  
[Cutting-off redundant repeating generations </br> for neural abstractive summarization](https://aclanthology.info/pdf/E/E17/E17-2047.pdf) | <ul><li>DUC-2004</li><li>Gigaword</li></ul> | <ul><li>DUC-2004</li><ul><li> ROUGE-1: **32.28** </li><li> ROUGE-2: 10.54 </li><li>ROUGE-L: **27.80** </li></ul><li>Gigaword</li><ul><li> ROUGE-1: **36.30** </li><li> ROUGE-2: 17.31 </li><li>ROUGE-L: **33.88** </li></ul></ul> | NOT YET AVAILABLE | 2017
[Convolutional Sequence to Sequence](https://arxiv.org/pdf/1705.03122.pdf) | <ul><li>DUC-2004</li><li>Gigaword</li></ul> | <ul><li>DUC-2004</li><ul><li> ROUGE-1: 33.44 </li><li> ROUGE-2: **10.84** </li><li>ROUGE-L: 26.90 </li></ul><li>Gigaword</li><ul><li> ROUGE-1: 35.88 </li><li> ROUGE-2: 27.48 </li><li>ROUGE-L: 33.29 </li></ul></ul> | [PyTorch](https://github.com/facebookresearch/fairseq-py) | 2017


#### 8. Dependency Parsing

Research Paper | Datasets  | Metric | Source Code | Year  
------------ | ------------- | ------------ | ------------- | -------------  
[Globally Normalized Transition-Based Neural Networks](https://arxiv.org/pdf/1603.06042.pdf) | <ul><li>Final CoNLL ’09 dependency parsing </li></ul> | <ul><li> 94.08% UAS accurancy</li> <li>92.15% LAS accurancy</li></ul> | <ul><li>[SyntaxNet](https://github.com/tensorflow/models/tree/master/research/syntaxnet) </li></ul>| <ul><li>2017</li></ul>


### Computer Vision

#### 1. Classification

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
      <td><a href='https://arxiv.org/pdf/1710.09829.pdf'> Dynamic Routing Between Capsules </a></td>
      <td align="left"> <ul><li> MNIST </li></ul> </td>
      <td align="left"> <ul><li> Test Error: 0.25±0.005 </li></ul> </td>
      <td align="left"> <ul><li>  <a href='https://github.com/Sarasra/models/tree/master/research/capsules'>Official Implementation</a> </li><li> <a href='https://github.com/gram-ai/capsule-networks'>PyTorch</a> </li><li> <a href='https://github.com/naturomics/CapsNet-Tensorflow'>Tensorflow</a> </li><li> <a href='https://github.com/XifengGuo/CapsNet-Keras'>Keras</a> </li><li>  <a href='https://github.com/soskek/dynamic_routing_between_capsules'>Chainer</a> </li> <li>  <a href='https://github.com/loretoparisi/CapsNet'>List of all implementations</a> </li>
          </ul>  </td>
      <td align="left">2017</td>    
    </tr>
    <tr>
      <td><a href='https://arxiv.org/pdf/1102.0183.pdf'> High-Performance Neural Networks for Visual Object Classification </a></td>
      <td align="left"> <ul><li> NORB </li></ul></td>
      <td align="left"> <ul><li> Test Error: 2.53 ± 0.40 </li></ul> </td>
      <td align="left"> <ul><li><a href=''>NOT FOUND</a></ul></li> </td>
      <td align="left">2011</td>    
    </tr>
    <tr>
      <td><a href='https://arxiv.org/pdf/1811.06965.pdf'>Giant AmoebaNet with GPipe</a></td>
      <td align="left"> <ul><li> CIFAR-10 </li> <li> CIFAR-100</li><li> ImageNet-1k</li><li> ...</li></ul></td>
      <td align="left"> <ul><li> Test Error: 1.0% </li> <li> Test Error: 8.7% </li><li> Top-1 Error 15.7</li><li> ...</li></ul> </td>
      <td align="left"> <ul><li> <a href=''>NOT FOUND</a> </li></ul> </td>
      <td align="left">2018</td>    
    </tr>
    <tr>
      <td><a href='https://openreview.net/pdf?id=S1NHaMW0b'>ShakeDrop regularization </a></td>
      <td align="left"> <ul><li> CIFAR-10 </li> <li> CIFAR-100</li></ul></td>
      <td align="left"> <ul><li> Test Error: 2.31% </li> <li> Test Error: 12.19% </li></ul> </td>
      <td align="left"> <ul><li> <a href=''>NOT FOUND</a> </li></ul> </td>
      <td align="left">2017</td>    
    </tr>
    <tr>
      <td><a href='https://arxiv.org/pdf/1611.05431.pdf'>Aggregated Residual Transformations for Deep Neural Networks </a></td>
      <td align="left"> <ul><li>  CIFAR-10  </li></ul></td>
      <td align="left"> <ul><li> Test Error: 3.58% </li></ul> </td>
      <td align="left"> <ul><li>  <a href='https://github.com/facebookresearch/ResNeXt'>PyTorch</a> </li></ul> </td>
      <td align="left">2017</td>    
    </tr>
    <tr>
      <td><a href='https://arxiv.org/abs/1708.04896'> Random Erasing Data Augmentation </a></td>
      <td align="left"> <ul><li> CIFAR-10 </li> <li> CIFAR-100 </li> <li> Fashion-MNIST </li> </ul></td>
      <td align="left"> <ul><li> Test Error: 3.08% </li>
          <li> Test Error: 17.73% </li>
          <li> Test Error: 3.65% </li>
          </ul> </td>
      <td align="left"> <a href='https://github.com/zhunzhong07/Random-Erasing'> Pytorch </td>
      <td align="left">2017</td>    
    </tr>
    <tr>
      <td><a href='https://arxiv.org/abs/1709.07634'> EraseReLU: A Simple Way to Ease the Training of Deep Convolution Neural Networks </a></td>
      <td align="left"> <ul><li> CIFAR-10 </li> <li> CIFAR-100 </li> </ul></td>
      <td align="left"> <ul><li> Test Error: 3.56% </li>
          <li> Test Error: 16.53% </li>
          </ul> </td>
      <td align="left"> <a href='https://github.com/D-X-Y/EraseReLU'> Pytorch </td>
      <td align="left">2017</td>    
    </tr>
    <tr>
      <td><a href='https://arxiv.org/pdf/1710.09829.pdf'> Dynamic Routing Between Capsules </a></td>
      <td align="left"> <ul><li> MultiMNIST </li></ul></td>
      <td align="left"> <ul><li> Test Error: 5% </li></ul> </td>
      <td align="left"> <ul><li> <a href='https://github.com/gram-ai/capsule-networks'>PyTorch</a> </li><li> <a href='https://github.com/naturomics/CapsNet-Tensorflow'>Tensorflow</a> </li><li> <a href='https://github.com/XifengGuo/CapsNet-Keras'>Keras</a> </li><li>  <a href='https://github.com/soskek/dynamic_routing_between_capsules'>Chainer</a> </li><li>  <a href='https://github.com/loretoparisi/CapsNet'>List of all implementations</a> </li></ul> </td>
      <td align="left">2017</td>    
    </tr>
    <tr>
      <td><a href='https://arxiv.org/pdf/1707.07012.pdf'>Learning Transferable Architectures for Scalable Image Recognition</a></td>
      <td align="left"> <ul><li>   ImageNet-1k  </li></ul></td>
      <td align="left"> <ul><li> Top-1 Error:17.3 </li></ul> </td>
      <td align="left"> <ul><li>  <a href='https://github.com/tensorflow/models/tree/master/research/slim/nets/nasnet'>Tensorflow</a> </li></ul> </td>
      <td align="left">2017</td>    
    </tr>
     <tr>
      <td><a href='https://arxiv.org/pdf/1709.01507.pdf'>Squeeze-and-Excitation Networks </a></td>
      <td align="left"> <ul><li>   ImageNet-1k  </li></ul></td>
      <td align="left"> <ul><li> Top-1 Error: 18.68 </li></ul> </td>
      <td align="left"> <ul><li>  <a href='https://github.com/hujie-frank/SENet'>CAFFE</a> </li></ul> </td>
      <td align="left">2017</td>    
    </tr>
    <tr>
      <td><a href='https://arxiv.org/pdf/1611.05431.pdf'>Aggregated Residual Transformations for Deep Neural Networks </a></td>
      <td align="left"> <ul><li>   ImageNet-1k  </li></ul></td>
      <td align="left"> <ul><li> Top-1 Error: 20.4% </li></ul> </td>
      <td align="left"> <ul><li>  <a href='https://github.com/facebookresearch/ResNeXt'>Torch</a> </li></ul> </td>
      <td align="left">2016</td>    
    </tr>
  </tbody>
</table>

#### 2. Instance Segmentation
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
      <td><a href='https://arxiv.org/pdf/1703.06870.pdf'>Mask R-CNN</a></td>
      <td align="left"> <ul><li> COCO  </li></ul></td>
      <td align="left"> <ul><li> Average Precision: 37.1% </li></ul> </td>
      <td align="left"> <ul><li>  <a href='https://github.com/facebookresearch/Detectron'>Detectron (Official Version)</a> </li><li>  <a href='https://github.com/TuSimple/mx-maskrcnn'>MXNet</a> </li><li>  <a href='https://github.com/matterport/Mask_RCNN'>Keras</a> </li><li>  <a href='https://github.com/CharlesShang/FastMaskRCNN'>TensorFlow </a> </li></ul> </td>
      <td align="left">2017</td>    
    </tr>
  </tbody>
</table>

#### 3. Visual Question Answering
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
      <td><a href='https://arxiv.org/abs/1708.02711'>Tips and Tricks for Visual Question Answering: Learnings from the 2017 Challenge</a></td>
      <td align="left"> <ul><li> VQA  </li></ul></td>
      <td align="left"> <ul><li> Overall score: 69 </li></ul> </td>
      <td align="left"> <ul><li>   <a href=''>NOT FOUND</a> </li></ul> </li></ul> </td>
      <td align="left">2017</td>    
    </tr>
  </tbody>
</table>

#### 4. Person Re-identification
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
      <td><a href='https://arxiv.org/abs/1708.04896'> Random Erasing Data Augmentation </a></td>
      <td align="left"> <ul><li> <a href='http://www.liangzheng.org/Project/project_reid.html'> Market-1501 </a> </li> <li> <a href='https://github.com/zhunzhong07/person-re-ranking'> CUHK03-new-protocol </a> </li> <li> <a href='https://github.com/layumi/DukeMTMC-reID_evaluation'> DukeMTMC-reID </a> </li> </ul></td>
      <td align="left"> <ul><li> Rank-1: 89.13 mAP: 83.93 </li>
          <li> Rank-1: 84.02 mAP: 78.28 </li>
          <li> labeled (Rank-1: 63.93 mAP: 65.05) detected (Rank-1: 64.43 mAP: 64.75) </li>
          </ul> </td>
      <td align="left"> <a href='https://github.com/zhunzhong07/Random-Erasing'> Pytorch </td>
      <td align="left">2017</td>    
    </tr>
  </tbody>
</table>

### Speech
[Speech SOTA](https://github.com/syhw/wer_are_we)
#### 1. ASR

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
      <td><a href='https://arxiv.org/pdf/1708.06073.pdf'>The Microsoft 2017 Conversational Speech Recognition System</a></td>
      <td align="left"> <ul><li> Switchboard Hub5'00  </li></ul></td>
      <td align="left"> <ul><li> WER: 5.1  </li></ul> </td>
      <td align="left"> <ul><li>  <a href=''>NOT FOUND</a></li></ul> </td>
      <td align="left">2017</td>    
    </tr>
    <tr>
      <td><a href='https://arxiv.org/pdf/1801.00059.pdf'>The CAPIO 2017 Conversational Speech Recognition System</a></td>
      <td align="left"> <ul><li> Switchboard Hub5'00  </li></ul></td>
      <td align="left"> <ul><li> WER: 5.0  </li></ul> </td>
      <td align="left"> <ul><li>  <a href=''>NOT FOUND</a></li></ul> </td>
      <td align="left">2017</td>    
    </tr>
  </tbody>
</table>


## Semi-supervised Learning
#### Computer Vision
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
      <td><a href='https://arxiv.org/pdf/1507.00677.pdf'> DISTRIBUTIONAL SMOOTHINGWITH VIRTUAL ADVERSARIAL TRAINING </a></td>
      <td align="left"> <ul><li> SVHN </li><li> NORB </li></ul></td>
      <td align="left"> <ul><li> Test error: 24.63 </li><li> Test error: 9.88 </li></ul> </td>
      <td align="left"> <a href='https://github.com/takerum/vat'>Theano</a></td>
      <td align="left">2016</td>    
    </tr>
     <tr>
      <td><a href='https://arxiv.org/pdf/1704.03976.pdf'> Virtual Adversarial Training:
a Regularization Method for Supervised and
Semi-supervised Learning </a></td>
      <td align="left"> <ul><li> MNIST </li></ul></td>
      <td align="left"> <ul><li> Test error: 1.27 </li></ul> </td>
      <td align="left"> <ul><li><a href=''>NOT FOUND</a></ul></li> </td>
      <td align="left">2017</td>    
    </tr>
    <tr>
      <td><a href='https://arxiv.org/pdf/1706.08249.pdf'> Few Shot Object Detection </a></td>
      <td align="left"> <ul><li> VOC2007 </li><li> VOC2012 </li></ul></td>
      <td align="left"> <ul><li> mAP : 41.7 </li><li> mAP : 35.4 </li></ul> </td>
      <td align="left"> <ul><li><a href=''>NOT FOUND</a></ul></li> </td>
      <td align="left">2017</td>    
    </tr>
    <tr>
      <td><a href='https://arxiv.org/pdf/1701.07717.pdf'> Unlabeled Samples Generated by GAN
Improve the Person Re-identification Baseline in vitro </a></td>
      <td align="left"> <ul><li> <a href='http://www.liangzheng.org/Project/project_reid.html'> Market-1501 </a> </li> <li> CUHK-03 </li> <li> <a href='https://github.com/layumi/DukeMTMC-reID_evaluation'> DukeMTMC-reID </a> </li> <li> <a href='http://www.vision.caltech.edu/visipedia/CUB-200-2011.html'> CUB-200-2011 </a></li></ul></td>
      <td align="left"> <ul><li> Rank-1: 83.97 mAP: 66.07 </li>
          <li> Rank-1: 84.6 mAP: 87.4 </li>
          <li> Rank-1: 67.68 mAP: 47.13 </li>
          <li> Test Accuracy: 84.4 </li>
          </ul> </td>
      <td align="left"> <a href='https://github.com/layumi/Person-reID_GAN'> Matconvnet </td>
      <td align="left">2017</td>    
    </tr>
      
  </tbody>
</table>

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
      <td><a href='http://research.nvidia.com/sites/default/files/publications/karras2017gan-paper-v2.pdf'> PROGRESSIVE GROWING OF GANS FOR IMPROVED QUALITY, STABILITY, AND VARIATION </a></td>
       <td align="left">Unsupervised CIFAR 10</td>
      <td align="left">Inception score: 8.80 </td>
      <td align="left"> <a href='https://github.com/tkarras/progressive_growing_of_gans'>Theano</a></td>
      <td align="left">2017</td>    
    </tr>
  </tbody>
</table>

### NLP

#### Machine Translation


<table>
  <tbody>
    <tr>
      <th width="30%">Research Paper</th>
      <th align="center" width="20%">Datasets</th>
      <th align="center" width="20%">Metric</th>
      <th align="center" width="20%">Source Code</th>
      <th align="center" width="10%">Year</th>
       <tr> 
      <td><a href='https://arxiv.org/pdf/1711.00043.pdf'>UNSUPERVISED MACHINE TRANSLATION
USING MONOLINGUAL CORPORA ONLY</a></td>
      <td align="left"> <ul><li> Multi30k-Task1(en-fr fr-en de-en en-de)  </li></ul></td>
      <td align="left"> <ul><li> BLEU:(32.76 32.07 26.26 22.74) </li></ul> </td>
      <td align="left"><ul><li><a href=''>NOT FOUND</a></ul></li></td>
      <td align="left">2017</td>    
    </tr>
        <tr> 
      <td><a href='https://arxiv.org/pdf/1804.09057.pdf'>Unsupervised Neural Machine Translation with Weight Sharing</a></td>
      <td align="left"> <ul><li> WMT14(en-fr fr-en)  </li><li> WMT16 (de-en en-de) </li></ul></td>
      <td align="left"> <ul><li> BLEU:(16.97 15.58) </li> <li> BLEU:(14.62 10.86) </li></ul> </td>
      <td align="left"><ul><li><a href=''>NOT FOUND</a></ul></li></td>
      <td align="left">2018</td>    
    </tr>
     

  </tbody>
</table>  

## Transfer Learning

<table>
  <tbody>
    <tr>
      <th width="30%">Research Paper</th>
      <th align="center" width="20%">Datasets</th>
      <th align="center" width="20%">Metric</th>
      <th align="center" width="20%">Source Code</th>
      <th align="center" width="10%">Year</th>
       <tr> 
      <td><a href='https://arxiv.org/pdf/1706.05137.pdf'>One Model To Learn Them All</a></td>
      <td align="left"> <ul><li> WMT EN → DE </li><li> WMT EN → FR (BLEU) </li><li> ImageNet (top-5 accuracy) </li></ul></td>
      <td align="left"> <ul><li> BLEU: 21.2 </li> <li> BLEU:30.5  </li><li> 86% </li></ul> </td>
      <td align="left"><ul><li><a href='https://github.com/tensorflow/tensor2tensor'>Tensorflow</a></ul></li></td>
      <td align="left">2017</td>    
    </tr>
      

  </tbody>
</table>  



## Reinforcement Learning
<table>
  <tbody>
    <tr>
      <th width="30%">Research Paper</th>
      <th align="center" width="20%">Datasets</th>
      <th align="center" width="20%">Metric</th>
      <th align="center" width="20%">Source Code</th>
      <th align="center" width="10%">Year</th>
       <tr> 
      <td><a href='http://www.gwern.net/docs/rl/2017-silver.pdf'>Mastering the game of Go without human knowledge</a></td>
      <td align="left"> the game of Go </td>
      <td align="left"> ElO Rating: 5185</td>
      <td align="left"><ul><li><a href=https://github.com/gcp/leela-zero>C++</a></ul></li></td>
      <td align="left">2017</td>    
    </tr>
      

  </tbody>
</table>  

Email: yxt.stoaml@gmail.com
