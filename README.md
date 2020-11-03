基于GAN的口令猜测漫步攻击 
GAN-based password guessing (trawling attacking)
=========================================================

## 基本介绍 Introduction
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
在巨人的肩膀上（基于Attribution中各个项目的代码），本项目进行了再创造，实现了两个基于GAN的口令猜测攻击模型，两个模型共同点和区别如下：

+ 共同点：
使用IWGAN，基于普通GAN，在判别器的损失函数中增加了一项约束项，具体可见相关论文。

+ 不同点：
生成器和判别器使用了不同的网络结构。
  1. 生成器和判别器使用5层ResNets。ResNets是一种基于CNN的Residual Blocks，进行残差学习。详见reference中的`PassGAN, A Deep Learning Approach for Password Guessing.pdf`。

  2. 生成器和判别器使用两层GRU循环神经网络。在这种情况下生成器。CNN是捕捉样本局部特征的有利工具，非常适合CV任务。但是口令猜测任务类似NLP的文本生成任务，一般使用循环神经网络，以利用其处理序列样本的优势。详见reference中的`Recurrent GANs Password Cracker For IoT Password Security Enhancement.pdf`。

## 使用方法 Usage
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
`train.py`用于训练模型，`sample.py`用于生成口令。
### 直接生成口令 Generate passwords
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
使用预训练好的模型，直接生成口令的方法如下（以csdn数据集为例）：
```bash
# 使用CNN-GAN
python sample.py -i ./csdn_cnn_model -o ./guess/cnn_gen_csdn.txt -m cnn -ck ./csdn_cnn_model/checkpoints/checkpoint_100000.ckpt
# 使用RNN-GAN
python sample.py -i ./csdn_rnn_model -o ./guess/rnn_gen_csdn.txt -m rnn -ck ./csdn_rnn_model/checkpoints/checkpoint_100000.ckpt
```
### 从头开始训练模型 Train from beginning
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
使用csdn数据集训练模型：
```bash
# 使用CNN-GAN
python train.py -i ./data/csdn_train.txt -o ./csdn_cnn_model -m cnn -b 64
# 使用RNN-GAN
python train.py -i ./data/csdn_train.txt -o ./csdn_rnn_model -m rnn -b 64
```
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
可以使用自己的数据集，改变-i参数即可。
### 继续训练模型 Restore training
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
比从头开始训练多了-ck命令行参数：
```bash
# 使用CNN-GAN
python train.py -i ./data/csdn_train.txt -o ./model/csdn_cnn_model -m cnn -b 64 -ck ./csdn_cnn_model/checkpoints/checkpoint_100000.ckpt
# 使用RNN-GAN
python train.py -i ./data/csdn_train.txt -o ./model/csdn_rnn_model -m rnn -b 64 -s 5000 -c 1 -ck ./csdn_cnn_model/checkpoints/checkpoint_100000.ckpt
```

## 其他问题 Other question
### python依赖要求
```bash
# requires CUDA 8 to be pre-installed
pip install -r requirements.txt
```
### 训练集和测试集
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
csdn数据集打乱后按照80%和20%比例划分为`csdn_train.txt`和`csdn_test.txt`，分别为训练集和测试集。yahoo数据集同理。模型训练均只使用训练集。

### 

## Attribution and License

This code is released under an [MIT License](https://github.com/brannondorsey/PassGAN/blob/master/LICENSE) and another [MIT License](https://github.com/igul222/improved_wgan_training/blob/master/LICENSE). You are free to use, modify, distribute, or sell it under those terms. 

The majority of the credit for the code in this repository goes to @brannondorsey for his work on the [PassGAN](https://github.com/brannondorsey/PassGAN), whose code is based on [improved_wgan_training](https://github.com/igul222/improved_wgan_training) written by @igul222. I've simply modified some module, added rnn-based implementation and changed some command-line interface.

The PassGAN [research and paper](https://arxiv.org/abs/1709.00440) was published by Briland Hitaj, Paolo Gasti, Giuseppe Ateniese, Fernando Perez-Cruz. The PassGAN based on RNN [research and paper](https://www.mdpi.com/1424-8220/20/11/3106) was published by Sungyup Nam, Seungho Jeon, Hongkyo Kim and Jongsub Moon. The default RNN-based PassGAN implemented in this project is different from above article, but you may change the neural network structure by providing specific command-line arguments.
