# -*- coding: utf-8 -*-
"""
Training a classifier
=====================

This is it. You have seen how to define neural networks, compute loss and make
updates to the weights of the network.

Now you might be thinking,

What about data?
----------------

Generally, when you have to deal with image, text, audio or video data,
you can use standard python packages that load data into a numpy array.
Then you can convert this array into a ``torch.*Tensor``.

-  For images, packages such as Pillow, OpenCV are useful.
-  For audio, packages such as scipy and librosa
-  For text, either raw Python or Cython based loading, or NLTK and
   SpaCy are useful.

Specifically for ``vision``, we have created a package called
``torchvision``, that has data loaders for common datasets such as
Imagenet, CIFAR10, MNIST, etc. and data transformers for images, viz.,
``torchvision.datasets`` and ``torch.utils.data.DataLoader``.

This provides a huge convenience and avoids writing boilerplate code.

For this tutorial, we will use the CIFAR10 dataset.
It has the classes: ‘airplane’, ‘automobile’, ‘bird’, ‘cat’, ‘deer’,
‘dog’, ‘frog’, ‘horse’, ‘ship’, ‘truck’. The images in CIFAR-10 are of
size 3x32x32, i.e. 3-channel color images of 32x32 pixels in size.

.. figure:: /_static/img/cifar10.png
   :alt: cifar10

   cifar10


Training an image classifier
----------------------------

We will do the following steps in order:

1. Load and normalizing the CIFAR10 training and test datasets using
   ``torchvision``
2. Define a Convolution Neural Network
3. Define a loss function
4. Train the network on the training data
5. Test the network on the test data

1. Loading and normalizing CIFAR10
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Using ``torchvision``, it’s extremely easy to load CIFAR10.
"""
import os
import math
import argparse
import datetime
import tqdm
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import six
import sys
from torch.optim.sign_internal_decay import LinearInternalDecay, CosineInternalDecay, RestartCosineInternalDecay



def main(args):
    ########################################################################
    # The output of torchvision datasets are PILImage images of range [0, 1].
    # We transform them to Tensors of normalized range [-1, 1]

    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    batch_size = args.batch_size
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=0) # orig num_workers:2

    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=0)

    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    ########################################################################
    # Let us show some of the training images, for fun.
    # functions to show an image
    def imshow(img):
        img = img / 2 + 0.5     # unnormalize
        npimg = img.numpy()
        plt.imshow(np.transpose(npimg, (1, 2, 0)))

    # get some random training images
    dataiter = iter(trainloader)
    images, labels = dataiter.next()

    # show images
    imshow(torchvision.utils.make_grid(images))
    # print labels
    print(' '.join('%5s' % classes[labels[j]] for j in range(4)))


    ########################################################################
    # 2. Define a Convolution Neural Network
    # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    # Copy the neural network from the Neural Networks section before and modify it to
    # take 3-channel images (instead of 1-channel images as it was defined).
    class Net(nn.Module):
        def __init__(self, n_conv_feat1=6, n_conv_feat2=16):
            super(Net, self).__init__()
            self.n_conv_feat1 = n_conv_feat1
            self.n_conv_feat2 = n_conv_feat2
            self.conv1 = nn.Conv2d(3, self.n_conv_feat1, 5)
            self.pool = nn.MaxPool2d(2, 2)
            self.conv2 = nn.Conv2d(self.n_conv_feat1, self.n_conv_feat2, 5)
            self.fc1 = nn.Linear(self.n_conv_feat2 * 5 * 5, 120)
            self.fc2 = nn.Linear(120, 84)
            self.fc3 = nn.Linear(84, 10)

        def forward(self, x):
            x = self.pool(F.relu(self.conv1(x)))
            x = self.pool(F.relu(self.conv2(x)))
            x = x.view(-1, self.n_conv_feat2 * 5 * 5)
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = self.fc3(x)
            return x

    net = Net()
    net = net.cuda()

    ########################################################################
    # 3. Define a Loss function and optimizer
    # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    # Let's use a Classification Cross-Entropy loss and SGD with momentum

    T_max = args.n_epochs * len(trainloader)
    decay_fn = None
    if args.sign_optim_decay == "linear":
        decay_fn = LinearInternalDecay(T_max)
    elif args.sign_optim_decay == "cosine":
        decay_fn = CosineInternalDecay(T_max, num_periods=args.sign_optim_decay_n_periods)
    elif args.sign_optim_decay == "restart":
        decay_fn = RestartCosineInternalDecay(T_max, num_periods=args.sign_optim_decay_n_periods)

    criterion = nn.CrossEntropyLoss()
    if args.optimizer == "sgd":
        optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9)
    elif args.optimizer == "adam":
        optimizer = optim.Adam(net.parameters(), lr=args.lr)
    elif args.optimizer == "add_sign":
        optimizer = optim.AddSign(net.parameters(), lr=args.lr, sign_internal_decay=decay_fn)
    elif args.optimizer == "power_sign":
        optimizer = optim.PowerSign(net.parameters(), lr=args.lr, sign_internal_decay=decay_fn)

    ########################################################################
    # 4. Train the network
    # ^^^^^^^^^^^^^^^^^^^^
    #
    # This is when things start to get interesting.
    # We simply have to loop over our data iterator, and feed the inputs to the
    # network and optimize

    iteration = 0
    now = datetime.datetime.now()
    print('Start Training at {}'.format(now.strftime('%Y/%m/%d-%H:%M:%S')))

    for epoch in range(args.n_epochs):  # 2 loop over the dataset multiple times
        running_loss = 0.0
        for i, data in tqdm.tqdm(enumerate(trainloader, 0), total=len(trainloader),
                    desc='Train iteration=%d' % iteration, ncols=80, leave=False):

            # get the inputs
            inputs, labels = data

            # wrap them in Variable
            inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.data[0]
            if (iteration + 1) % args.stat_interval == 0:
                step, internal_lr = 1, 0
                if args.optimizer not in ["sgd", "adam"]:
                    step = six.next(six.itervalues(optimizer.state))["step"]
                    internal_lr = optimizer.param_groups[0]['sign_internal_decay'](step - 1)

                print('\n[{:3d}, {:5d}, {:6d}] loss: {:.3f}, lr: {:.6f}, step: {:6d}'.format(epoch + 1, i + 1, iteration + 1, running_loss / args.stat_interval, internal_lr, step - 1))
                running_loss = 0.0

            iteration += 1

        evaluate(net, testloader, classes, epoch, args)


    now = datetime.datetime.now()
    print('Finished Training at {}'.format(now.strftime('%Y/%m/%d-%H:%M:%S')))

    ########################################################################
    # 5. Test the network on the test data
    # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    #
    # We have trained the network for 2 passes over the training dataset.
    # But we need to check if the network has learnt anything at all.
    #
    # We will check this by predicting the class label that the neural network
    # outputs, and checking it against the ground-truth. If the prediction is
    # correct, we add the sample to the list of correct predictions.
    #
    # Okay, first step. Let us display an image from the test set to get familiar.

    dataiter = iter(testloader)
    images, labels = dataiter.next()

    # print images
    imshow(torchvision.utils.make_grid(images))
    print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(4)))

    ########################################################################
    # Okay, now let us see what the neural network thinks these examples above are:

    outputs = net(Variable(images.cuda()))

    ########################################################################
    # The outputs are energies for the 10 classes.
    # Higher the energy for a class, the more the network
    # thinks that the image is of the particular class.
    # So, let's get the index of the highest energy:
    _, predicted = torch.max(outputs.data, 1)

    print('Predicted: ', ' '.join('%5s' % classes[predicted[j]]
                                  for j in range(4)))

def evaluate(net, testloader, classes, epoch, args):
    ########################################################################
    # The results seem pretty good.
    #
    # Let us look at how the network performs on the whole dataset.

    correct = 0
    total = 0
    for data in testloader:
        images, labels = data
        labels = labels.cuda()
        outputs = net(Variable(images.cuda()))
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum()

    print('\nAccuracy of the network on the 10000 test images: {:.2f} %, epoch: {}, lr: {:.5f}'.format(100.0 * correct / total, epoch, args.lr))

    ########################################################################
    # That looks waaay better than chance, which is 10% accuracy (randomly picking
    # a class out of 10 classes).
    # Seems like the network learnt something.
    #
    # Hmmm, what are the classes that performed well, and the classes that did
    # not perform well:

    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))
    for data in testloader:
        images, labels = data
        labels = labels.cuda()
        outputs = net(Variable(images.cuda()))
        _, predicted = torch.max(outputs.data, 1)
        c = (predicted == labels).squeeze()
        for i in range(4):
            label = labels[i]
            class_correct[label] += c[i]
            class_total[label] += 1


    for i in range(10):
        print('Accuracy of {} : {:.1f} %, epoch: {}, lr: {:.5f}'.format(classes[i],
                                100 * class_correct[i] / class_total[i], epoch, args.lr))

if __name__ == '__main__':

    print("Python version: {}".format(sys.version))
    print("PyTorch version: {}".format(torch.__version__))

    parser = argparse.ArgumentParser()
    parser.add_argument('-g', '--gpu', type=int, default=0, help='')
    parser.add_argument('--batch_size', type=int, default=4, help='')
    # parser.add_argument('--n_epochs', type=int, default=20, help='')
    parser.add_argument('--n_epochs', type=int, default=1, help='')
    parser.add_argument('--n_conv_feat1', type=int, default=6, help='')
    parser.add_argument('--n_conv_feat2', type=int, default=16, help='')
    parser.add_argument('--optimizer', type=str, default="sgd", help='sgd/adam/add_sign/power_sign')
    parser.add_argument('--lr', type=float, default=0.001, help='')
    parser.add_argument('--sign_optim_decay', type=str, default=None,
                        help='linear/cosine/restart. Valid for AddSign/PowerSign optimizer')
    parser.add_argument('--sign_optim_decay_n_periods', type=float, default=0.5,
                        help='valid for cosine/restart internal decay')
    parser.add_argument('--stat_interval', type=int, default=100, help='')
    args = parser.parse_args()
    print(args)

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    args.cuda = torch.cuda.is_available()
    if args.cuda:
        print("torch.backends.cudnn.version: {}".format(torch.backends.cudnn.version()))

    main(args)

########################################################################
# Okay, so what next?
#
# How do we run these neural networks on the GPU?
#
# Training on GPU
# ----------------
# Just like how you transfer a Tensor on to the GPU, you transfer the neural
# net onto the GPU.
# This will recursively go over all modules and convert their parameters and
# buffers to CUDA tensors:
#
# .. code:: python
#
#     net.cuda()
#
#
# Remember that you will have to send the inputs and targets at every step
# to the GPU too:
#
# ::
#
#         inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
#
# Why dont I notice MASSIVE speedup compared to CPU? Because your network
# is realllly small.
#
# **Exercise:** Try increasing the width of your network (argument 2 of
# the first ``nn.Conv2d``, and argument 1 of the second ``nn.Conv2d`` –
# they need to be the same number), see what kind of speedup you get.
#
# **Goals achieved**:
#
# - Understanding PyTorch's Tensor library and neural networks at a high level.
# - Train a small neural network to classify images
#
# Training on multiple GPUs
# -------------------------
# If you want to see even more MASSIVE speedup using all of your GPUs,
# please check out :doc:`data_parallel_tutorial`.
#
# Where do I go next?
# -------------------
#
# -  :doc:`Train neural nets to play video games </intermediate/reinforcement_q_learning>`
# -  `Train a state-of-the-art ResNet network on imagenet`_
# -  `Train a face generator using Generative Adversarial Networks`_
# -  `Train a word-level language model using Recurrent LSTM networks`_
# -  `More examples`_
# -  `More tutorials`_
# -  `Discuss PyTorch on the Forums`_
# -  `Chat with other users on Slack`_
#
# .. _Train a state-of-the-art ResNet network on imagenet: https://github.com/pytorch/examples/tree/master/imagenet
# .. _Train a face generator using Generative Adversarial Networks: https://github.com/pytorch/examples/tree/master/dcgan
# .. _Train a word-level language model using Recurrent LSTM networks: https://github.com/pytorch/examples/tree/master/word_language_model
# .. _More examples: https://github.com/pytorch/examples
# .. _More tutorials: https://github.com/pytorch/tutorials
# .. _Discuss PyTorch on the Forums: https://discuss.pytorch.org/
# .. _Chat with other users on Slack: http://pytorch.slack.com/messages/beginner/
