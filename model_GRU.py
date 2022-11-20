import torch
from config import *
from torch import dropout, nn                   ##Import pytorch Neural Network Module
import torch.nn.functional as F         ##Import activation and loss functions

# If you’re using negative log likelihood loss and log softmax activation, 
# then Pytorch provides a single function F.cross_entropy that combines the two.

# nn.Module (uppercase M) is a PyTorch specific concept, and is a class we’ll be using a lot.
# nn.Module is not to be confused with the Python concept of a (lowercase m) module, 
# which is a file of Python code that can be imported.

class SCNN(nn.Module):
    def __init__(self):
        super().__init__()
        kernel = 9
        # Initialize convolution for down to up and up to down message passing
        self.up_down_conv = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(1,kernel), padding=(0,4))
        self.down_up_conv = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(1,kernel), padding=(0,4))
        self.left_right_conv = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(kernel,1), padding=(4,0))
        self.right_left_conv = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(kernel,1), padding=(4,0))
        self.scnn = nn.ModuleList([self.up_down_conv, self.down_up_conv, self.left_right_conv, self.right_left_conv])
        self.relu = nn.ReLU()


    def forward(self, x):

        batch, channels, height, width = x.size()

        ###up to down#####
        for i in range(1, height):
            x[:, :, i:i+1, :] = (x[:, :, i:i+1, :].clone()).add(self.relu(self.scnn[0](x[:, :, i-1:i, :].clone())))

        ###down to up###
        for i in range(height-2, -1, -1):
            x[:, :, i:i+1, :] = (x[:, :, i:i+1, :].clone()).add(self.relu(self.scnn[1](x[:, :, i+1:i+2, :].clone())))
        
        ###left to right###
        for i in range(1, width):
            x[:, :, :, i:i+1] = (x[:, :, :, i:i+1].clone()).add(self.relu(self.scnn[2](x[:, :, :, i-1:i].clone())))

        ###right to left###
        for i in range(width-2, -1, -1):
            x[:, :, :, i:i+1] = (x[:, :, :, i:i+1].clone()).add(self.relu(self.scnn[3](x[:, :, :, i+1:i+2].clone())))
        return x


class Conv_GRU(nn.Module):
    def __init__(self, input_channels, hidden_state_channels):
        super(Conv_GRU, self).__init__()
        self.input_channels = input_channels
        self.hidden_state_channels = hidden_state_channels
        self.conv_z = nn.Conv2d(in_channels=self.input_channels + self.hidden_state_channels, out_channels=self.hidden_state_channels\
                                , kernel_size=(3,3), stride=(1,1), padding=(1,1))
        
        self.conv_r = nn.Conv2d(in_channels=self.input_channels + self.hidden_state_channels, out_channels=self.hidden_state_channels\
                                , kernel_size=(3,3), stride=(1,1), padding=(1,1))
                                
        self.conv_h_ = nn.Conv2d(in_channels=self.input_channels + self.hidden_state_channels, out_channels=self.hidden_state_channels\
                                , kernel_size=(3,3), stride=(1,1), padding=(1,1))
        self.dropout = nn.Dropout(0.5)
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()

    def forward(self, hidden_state, input):
        # Input is of dimensions = minibatch, channels, height, width

        concatenate = torch.cat([hidden_state, input], dim=1)

        z = self.sigmoid(self.conv_z(concatenate))
        r = self.sigmoid(self.conv_r(concatenate))
        h_ = self.tanh(self.conv_h_(torch.cat([r * hidden_state, input], dim=1)))
        h = (1 - z) * hidden_state + z * h_

        return h

class RNN_ConvGRU(nn.Module):
    def __init__(self, input_channels, hidden_state_channels):
        super(RNN_ConvGRU, self).__init__()

        self.convgru = Conv_GRU(input_channels, hidden_state_channels)

    def forward(self, hidden_state, input_array):
        # input array is a list of tensors of dimensions = mini_batch, channels, height, width
        hidden_states = []
        hidden_states.append(hidden_state)
        for i in range(len(input_array)):
            h = self.convgru(hidden_states[i], input_array[i])
            hidden_states.append(h)
        hidden_states.pop(0)

        return hidden_states

class STRNN(nn.Module):                             ##Segnet Based nn
    def __init__(self, inChannels=3, outChannels=2):
        # call the parent constructor
        super(STRNN, self).__init__()

        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=(2,2), padding=(0,0), stride=2, return_indices= True)
        self.maxunpool = nn.MaxUnpool2d(kernel_size=(2,2), padding=(0,0), stride=2)
        self.batchnorm = nn.BatchNorm2d(64)
        ######## Encoder Architecture ##########


        # initialize first set of CONV => RELU => CONV => RELU => POOL layers
        self.en_conv_1_1 = nn.Conv2d(in_channels=inChannels, out_channels=64, kernel_size=(3,3), padding=(1,1), stride=1)
        self.en_conv_1_2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3,3), padding=(1,1), stride=1)
        self.batchnorm_1 = nn.BatchNorm2d(64)

        # initialize the SCNN layer
        self.scnn = SCNN()

        # initialize second set of CONV => RELU => CONV => RELU => POOL layers
        self.en_conv_2_1 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3,3), padding=(1,1), stride=1)
        self.en_conv_2_2 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3,3), padding=(1,1), stride=1)
        self.batchnorm_2 = nn.BatchNorm2d(128)

        # initialize third set of CONV => RELU => CONV => RELU => CONV => RELU => POOL layers
        self.en_conv_3_1 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3,3), padding=(1,1), stride=1)
        self.en_conv_3_2 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3,3), padding=(1,1), stride=1)
        self.en_conv_3_3 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3,3), padding=(1,1), stride=1)
        self.batchnorm_3 = nn.BatchNorm2d(256)

        # initialize fourth set of CONV => RELU => CONV => RELU => CONV => RELU => POOL layers
        self.en_conv_4_1 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(3,3), padding=(1,1), stride=1)
        self.en_conv_4_2 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3,3), padding=(1,1), stride=1)
        self.en_conv_4_3 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3,3), padding=(1,1), stride=1)
        self.batchnorm_4 = nn.BatchNorm2d(512)

        # initialize fifth set of CONV => RELU => CONV => RELU => CONV => RELU => POOL layers
        self.en_conv_5_1 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3,3), padding=(1,1), stride=1)
        self.en_conv_5_2 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3,3), padding=(1,1), stride=1)
        self.en_conv_5_3 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3,3), padding=(1,1), stride=1)
        self.batchnorm_5 = nn.BatchNorm2d(512)

        ####### Encoder Architecture Ends ##############

        ########## RNN Architecture ########

        # initialize the ConvLSTM module
        self.rnn_1 = RNN_ConvGRU(512, 512)
        self.rnn_2 = RNN_ConvGRU(512,512)

        ######### RNN Architecture Ends ###########
        
        ##################  Decoder Architecture  ###############

        # initialize first set of UNPOOL => CONV => RELU => CONV => RELU => CONV => RELU layers
        self.de_conv_5_1 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3,3), padding=(1,1), stride=1)
        self.de_conv_5_2 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3,3), padding=(1,1), stride=1)
        self.de_conv_5_3 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3,3), padding=(1,1), stride=1)

        # initialize second set of UNPOOL => CONV => RELU => CONV => RELU => CONV => RELU layers
        self.de_conv_4_1 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3,3), padding=(1,1), stride=1)
        self.de_conv_4_2 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3,3), padding=(1,1), stride=1)
        self.de_conv_4_3 = nn.Conv2d(in_channels=512, out_channels=256, kernel_size=(3,3), padding=(1,1), stride=1)

        # initialize third set of UNPOOL => CONV => RELU => CONV => RELU => CONV => RELU layers
        self.de_conv_3_1 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3,3), padding=(1,1), stride=1)
        self.de_conv_3_2 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3,3), padding=(1,1), stride=1)
        self.de_conv_3_3 = nn.Conv2d(in_channels=256, out_channels=128, kernel_size=(3,3), padding=(1,1), stride=1)

        # initialize fourth set of UNPOOL => CONV => RELU => CONV => RELU layers
        self.de_conv_2_2 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3,3), padding=(1,1), stride=1)
        self.de_conv_2_3 = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=(3,3), padding=(1,1), stride=1)

        # initialize fourth set of UNPOOL => CONV => RELU => CONV => RELU layers
        self.de_conv_1_2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3,3), padding=(1,1), stride=1)
        self.de_conv_1_3 = nn.Conv2d(in_channels=64, out_channels=2, kernel_size=(3,3), padding=(1,1), stride=1)
        self.log_softmax = nn.LogSoftmax(dim=1)

    ########## Decoder Architecture Ends ##########

    def block_1(self, input):
        x1 = self.en_conv_1_1(input)
        x1 = self.batchnorm_1(x1)
        x1 = self.relu(x1)
        # print(x1)
        x1 = self.en_conv_1_2(x1)
        x1 = self.batchnorm_1(x1)
        x1 = self.relu(x1)
        x1, indices_1 = self.maxpool(x1)
        return x1, indices_1

    def do_scnn(self, input):
        xscnn = self.scnn(input) 
        xscnn = self.batchnorm(xscnn)
        # print(xscnn)
        return xscnn

    def block_2(self, input):
        x2 = self.en_conv_2_1(input)
        x2 = self.batchnorm_2(x2)
        x2 = self.relu(x2)
        x2 = self.en_conv_2_2(x2)
        x2 = self.batchnorm_2(x2)
        x2 = self.relu(x2)
        x2, indices_2 = self.maxpool(x2)
        # print(x2)
        return x2, indices_2

    def block_3(self, input):
        x3 = self.en_conv_3_1(input)
        x3 = self.batchnorm_3(x3)
        x3 = self.relu(x3)
        x3 = self.en_conv_3_2(x3)
        x3 = self.batchnorm_3(x3)
        x3 = self.relu(x3)
        x3 = self.en_conv_3_3(x3)
        x3 = self.batchnorm_3(x3)
        x3 = self.relu(x3)
        x3, indices_3 = self.maxpool(x3)
        # print(x3)
        return x3, indices_3

    def block_4(self, input):
        x4 = self.en_conv_4_1(input)
        x4 = self.batchnorm_4(x4)
        x4 = self.relu(x4)
        x4 = self.en_conv_4_2(x4)
        x4 = self.batchnorm_4(x4)
        x4 = self.relu(x4)
        x4 = self.en_conv_4_3(x4)
        x4 = self.batchnorm_4(x4)
        x4 = self.relu(x4)
        x4, indices_4 = self.maxpool(x4)
        return x4, indices_4

    def block_5(self, input):
        x5 = self.en_conv_5_1(input)
        x5 = self.batchnorm_5(x5)
        x5 = self.relu(x5)
        x5 = self.en_conv_5_2(x5)
        x5 = self.batchnorm_5(x5)
        x5 = self.relu(x5)
        x5 = self.en_conv_5_3(x5)
        x5 = self.batchnorm_5(x5)
        x5 = self.relu(x5)
        x5, indices_5 = self.maxpool(x5)
        return x5, indices_5

    def block_6(self, input, indices_5):
        x6 = self.maxunpool(input, indices_5)
        x6 = self.de_conv_5_1(x6)
        x6 = self.batchnorm_5(x6)
        x6 = self.relu(x6)
        x6 = self.de_conv_5_2(x6)
        x6 = self.batchnorm_5(x6)
        x6 = self.relu(x6)
        x6 = self.de_conv_5_3(x6)
        x6 = self.batchnorm_5(x6)
        x6 = self.relu(x6)
        return x6

    def block_7(self, input, indices_4):
        x7 = self.maxunpool(input, indices_4)
        x7 = self.de_conv_4_1(x7)
        x7 = self.batchnorm_4(x7)
        x7 = self.relu(x7)
        x7 = self.de_conv_4_2(x7)
        x7 = self.batchnorm_4(x7)
        x7 = self.relu(x7)
        x7 = self.de_conv_4_3(x7)
        x7 = self.batchnorm_3(x7)
        x7 = self.relu(x7)
        return x7

    def block_8(self, input, indices_3):
        x8 = self.maxunpool(input, indices_3)
        x8 = self.de_conv_3_1(x8)
        x8 = self.batchnorm_3(x8)
        x8 = self.relu(x8)
        x8 = self.de_conv_3_2(x8)
        x8 = self.batchnorm_3(x8)
        x8 = self.relu(x8)
        x8 = self.de_conv_3_3(x8)
        x8 = self.batchnorm_2(x8)
        x8 = self.relu(x8)
        return x8

    def block_9(self, input, indices_2):
        x9 = self.maxunpool(input, indices_2)
        x9 = self.de_conv_2_2(x9)
        x9 = self.batchnorm_2(x9)
        x9 = self.relu(x9)
        x9 = self.de_conv_2_3(x9)
        x9 = self.batchnorm_1(x9)
        x9 = self.relu(x9)
        # print(x9)
        return x9

    def block_10(self, input, indices_1):
        x10 = self.maxunpool(input, indices_1)
        x10 = self.de_conv_1_2(x10)
        x10 = self.batchnorm_1(x10)
        x10 = self.relu(x10)
        x10 = self.de_conv_1_3(x10)
        x10 = self.log_softmax(x10)
        # print(x10)
        return x10

    def forward(self, input):
        # input has dimensions = mini_batch, 5, channels, Height, Width
        input_ = torch.tensor_split(input, 5, dim=1)
        feat = []
        for i in range(5):
            img = torch.squeeze(input_[i], dim=1)  # dimensions = mini_batch, channels, height, width
            # print(img)
            x1, indices_1 = self.block_1(img)
            # print(x1)
            xSCNN = self.do_scnn(x1)
            # print(xSCNN)
            x2, indices_2 = self.block_2(xSCNN)
            # print(x2)
            x3, indices_3 = self.block_3(x2)
            # print(x3)
            x4, indices_4 = self.block_4(x3)
            # print(x4)
            x5, indices_5 = self.block_5(x4)
            # print(x5)        
            feat.append(x5)
        shape = feat[0].size()
        hidden_init = torch.zeros(shape).to(torch.device(DEVICE))
        xRNN_1 = self.rnn_1(hidden_init, feat)
        # print(xRNN_1)
        xRNN_2 = self.rnn_2(hidden_init, xRNN_1)
        xRNN_out = xRNN_2[len(xRNN_2) - 1]       # output is the last element of final rnn layer
        # xRNN_out is of dimensions = mini_batch, channels, height, width
        x6 = self.block_6(xRNN_out, indices_5)
        x7 = self.block_7(x6, indices_4)
        x8 = self.block_8(x7, indices_3)
        x9 = self.block_9(x8, indices_2)
        x10 = self.block_10(x9, indices_1)
        return x10
        