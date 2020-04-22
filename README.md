# Installing and Using SAM-ResNet for Dummies

Simple, step-by-step instructions for setting up SAM-ResNet on Windows 10.

Author: Brenden Eum (Apr 2020)

## Introduction 

SAM-ResNet is a Saliency Attentive Model developed by Cornia, Baraldi, Serra, and Cucchiara (2018). Without diving into too many technicalities, SAM-ResNet utilizes a convolutional LSTM (long short term memory) neural network structure to predict saliency maps for images by incorporating neural attentive mechanisms. To learn more about the model, see the [paper](https://aimagelab.ing.unimore.it/imagelab/pubblicazioni/2018-tip.pdf).

As of April 21, 2020, SAM-ResNet is the top performing model on the MIT saliency benchmark test (cat2000). Rankings can be found [here](http://saliency.mit.edu/results_cat2000.html).

This document will provide simple step-by-step instructions to get you up and running with SAM-ResNet. Its purpose is not to get you acquainted with the theory behind the model. Instead, it will simply go over:
* How to install everything you need to run SAM-ResNet
* How to run SAM-ResNet using pre-trained models by Cornia, Baraldi, Serra, and Cucchiara (2018)
* How to train SAM-ResNet using your own images 
* How to use Python to adjust image characteristics (e.g. brightness, color saturation, etc.)

Everything will be done using Python and Command Prompt (PC) or terminal (Mac). The instructions here are for PC, and I will interchange between saying “command prompt” and “terminal”. This is not the only way to set up SAM-ResNet (and not the most efficient way), but if you are unfamiliar with setting up your GPU to use for deep learning, I find that this may be one of the simplest ways. Once you get acquainted with this process, you might find more advanced setups on your own that can lead to faster processing.

If you feel you do have a strong background in using terminals, you may be interested in using the instructions laid out in Cornia, Baraldi, Serra, and Cucchiara’s [README file](https://github.com/marcellacornia/sam). They are the original instructions followed by the author of this document, which assumes you are familiar with using terminals and are much more concise. The instructions found there are missing some steps needed for PC users.

## Installing

### 1 Setting up a toolkit folder

This first step is more of a stylistic choice, but I like to set aside a folder where I am going to install everything I need for deep learning. Preferably, you would place this in a drive with plenty of space (at least 30 GB of free space). For instance, I created a folder in my D: drive.

```
D:\toolkits.win
```

Let’s call this your *toolkit folder*.
