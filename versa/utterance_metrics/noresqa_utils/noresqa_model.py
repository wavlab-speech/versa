#Copyright (c) Meta Platforms, Inc. and affiliates.
#All rights reserved.

#This source code is licensed under the license found in the
#LICENSE file in the root directory of this source tree.


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm
import numpy as np
from librosa.filters import mel as librosa_mel_fn
from torch.nn import Parameter
from functools import wraps
import fairseq
from fairseq import tasks
import pickle

class model_dimred(nn.Module):

    def __init__(self, in_channel=64, conv1x1=16, reduce3x3=24, conv3x3=32, reduce5x5=16, conv5x5=8, pool_proj=8, pool=2):
        super(model_dimred, self).__init__()

        self.modules1 = nn.ModuleList()
        self.modules1.append(nn.Conv2d(in_channel, conv1x1, 1, (1,1), 0))
        self.modules1.append(nn.Conv2d(in_channel, reduce3x3, 1, 1, 0))
        self.modules1.append(nn.Conv2d(reduce3x3, conv3x3, 3, (1,1), 1))
        self.modules1.append(nn.Conv2d(in_channel, reduce5x5, 1, 1, 0))
        self.modules1.append(nn.Conv2d(reduce5x5, conv5x5, 5, (1,1), 2))
        self.modules1.append(nn.MaxPool2d((3,3),stride=(1,1),padding=(1,1)))
        self.modules1.append(nn.Conv2d(in_channel, pool_proj, 1, 1, 0))
        self.modules1.append(nn.MaxPool2d((1,pool)))

    def forward(self, x):

        a = F.relu(self.modules1[0](x))
        b = F.relu(self.modules1[2]((F.relu(self.modules1[1](x)))))
        c = F.relu(self.modules1[4]((F.relu(self.modules1[3](x)))))
        d = F.relu(self.modules1[5](x))
        d = F.relu(self.modules1[6](d))
        x1 = torch.cat((a, b, c, d), axis=1)
        x2 = F.relu(self.modules1[7](x1))
        return x2


class base_encoder(nn.Module):
    def __init__(self,dev=torch.device('cpu')):
        super(base_encoder, self).__init__()
        self.dev = dev

        self.modelA = model_dimred(in_channel=2, pool=4)
        self.modelB = model_dimred(in_channel=64, pool=4)
        self.modelC = model_dimred(in_channel=64, pool=4)
        self.modelD = model_dimred(in_channel=64, pool=2)


    def forward(self,x):
        x = (self.modelD(self.modelC(self.modelB(self.modelA(x)))))
        return x


class which_clean(nn.Module):
    def __init__(self):
        super(which_clean, self).__init__()
        n_layers = 2

        self.encoder = nn.ModuleList()
        self.ebatch = nn.ModuleList()
        self.dp = nn.ModuleList()
        filter_size = 5
        dp_num = 0.50
        self.encoder.append(nn.Conv1d(128,32,filter_size,padding=filter_size//2))
        self.ebatch.append(nn.BatchNorm1d(32))
        self.dp.append(nn.Dropout(p=dp_num))
        self.encoder.append(nn.Conv1d(32,8,filter_size,padding=filter_size//2))
        self.ebatch.append(nn.BatchNorm1d(8))
        self.dp.append(nn.Dropout(p=dp_num))
        self.encoder.append(nn.Conv1d(8,2,filter_size,padding=filter_size//2))
        self.ebatch.append(nn.BatchNorm1d(2))
        self.dp.append(nn.Dropout(p=dp_num))


    def forward(self,x):

        for i in range(3):
            x = self.encoder[i](x)
            x = self.ebatch[i](x)
            if i!=2:
                x = F.leaky_relu(x,0.1)
            x = self.dp[i](x)
        return x

class how_snr(nn.Module):
    def __init__(self,dim_emb=32, output=50):
        super(how_snr, self).__init__()
        n_layers = 2

        self.encoder = nn.ModuleList()
        self.ebatch = nn.ModuleList()
        self.dp = nn.ModuleList()
        filter_size = 5
        dp_num = 0.50
        self.encoder.append(nn.Conv1d(128,64,filter_size,padding=filter_size//2))
        self.ebatch.append(nn.BatchNorm1d(64))
        self.dp.append(nn.Dropout(p=dp_num))
        self.encoder.append(nn.Conv1d(64,32,filter_size,padding=filter_size//2))
        self.ebatch.append(nn.BatchNorm1d(32))
        self.dp.append(nn.Dropout(p=dp_num))
        self.encoder.append(nn.Conv1d(32,output,filter_size,padding=filter_size//2))
        self.ebatch.append(nn.BatchNorm1d(output))
        self.dp.append(nn.Dropout(p=dp_num))

    def forward(self,x):

        for i in range(3):
            x = self.encoder[i](x)
            x = self.ebatch[i](x)
            if i!=2:
                x = F.leaky_relu(x,0.1)
            x = self.dp[i](x)
        return x

class how_snr_snr(nn.Module):
    def __init__(self,dim_emb=32, output=50):
        super(how_snr_snr, self).__init__()
        n_layers = 2

        self.encoder = nn.ModuleList()
        self.ebatch = nn.ModuleList()
        self.dp = nn.ModuleList()
        filter_size = 5
        dp_num = 0.50
        self.encoder.append(nn.Conv1d(128,64,filter_size,padding=filter_size//2))
        self.ebatch.append(nn.BatchNorm1d(64))
        self.dp.append(nn.Dropout(p=dp_num))
        self.encoder.append(nn.Conv1d(64,32,filter_size,padding=filter_size//2))
        self.ebatch.append(nn.BatchNorm1d(32))
        self.dp.append(nn.Dropout(p=dp_num))
        self.encoder.append(nn.Conv1d(32,output,filter_size,padding=filter_size//2))
        self.ebatch.append(nn.BatchNorm1d(output))
        self.dp.append(nn.Dropout(p=dp_num))

    def forward(self,x):

        for i in range(3):
            x = self.encoder[i](x)
            x = self.ebatch[i](x)
            if i!=2:
                x = F.leaky_relu(x,0.1)
            x = self.dp[i](x)
        return x

class NORESQA(nn.Module):

    def __init__(self,dev=torch.device('cpu'), minit=1, output=20,output2=16, metric_type=0, config_path='models/wav2vec_small.pt'):
        super(NORESQA, self).__init__()

        self.metric_type = metric_type
        if metric_type==0:
            self.base_encoder = base_encoder()
            self.base_encoder_2 = TemporalConvNet(num_inputs=128,num_channels=[32,64,128,64],kernel_size=3)

            self.which_clean = which_clean()
            self.how_snr_sdr = how_snr(output=output)
            self.how_snr_snr = how_snr_snr(output=output2)
            if minit == 1:
                self.base_encoder.apply(weights_init)
                self.which_clean.apply(weights_init)
                self.how_snr_sdr.apply(weights_init)
                self.how_snr_snr.apply(weights_init)
            self.CE = nn.CrossEntropyLoss(reduction='mean')

        elif metric_type == 1:

            SSL_OUT_DIM=768
            ssl_model, _, _ = fairseq.checkpoint_utils.load_model_ensemble_and_task([config_path])

            ssl_model = ssl_model[0]

            ssl_model.remove_pretraining_modules()
            self.main_model = MosPredictor(ssl_model, SSL_OUT_DIM)
            self.linear_layer = nn.Linear(SSL_OUT_DIM, 32)

            self.quantification = PoolAtt(d_input=64,output_size=5)
            self.preference = PoolAtt(d_input=64,output_size=2)


    def forward(self, x1, x2 = None):

        if self.metric_type == 0:
            x1 = self.base_encoder.forward(x1)
            x2 = self.base_encoder.forward(x2)
            x1=self.base_encoder_2(x1)
            x2=self.base_encoder_2(x2)

            concat = torch.cat((x1,x2), 1)

            which_closer = self.which_clean.forward(concat)
            sdr_diff = self.how_snr_sdr.forward(concat)
            snr_diff = self.how_snr_snr.forward(concat)

            return which_closer, sdr_diff, snr_diff

        elif self.metric_type == 1:

            x1 = self.linear_layer(self.main_model(x1)).permute(0,2,1)
            y1 = self.linear_layer(self.main_model(x2)).permute(0,2,1)
            concat = torch.cat((x1,y1), 1)

            n_wins = concat.shape[2]
            B = [n_wins for n in range(concat.shape[0])]
            n_wins_tensor = torch.from_numpy(np.asarray(B)).to(concat.device)

            pref = self.preference(concat.permute(0,2,1),n_wins_tensor)
            quantf = self.quantification(concat.permute(0,2,1),n_wins_tensor)

            att = F.softmax(quantf, dim=1)
            B = torch.linspace(0, 4, steps=5).to(concat.device)
            C = (att*B).sum(axis=1)
            return C


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1 or classname.find('BatchNorm') != -1 or classname.find('Linear') != -1:
        torch.nn.init.normal_(m.weight)
        try:
            torch.nn.init.constant_(m.bias, 0.01)
        except:
            pass



class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=dilation, dilation=dilation))

        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=dilation, dilation=dilation))
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.relu1, self.dropout1,
                                 self.conv2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size, dropout=dropout)]

        self.network = nn.Sequential(*layers)

    def forward(self, x1):

        x1 = x1.reshape(x1.shape[0],-1,x1.shape[2])
        x = self.network(x1)
        return x


class MosPredictor(nn.Module):
    def __init__(self, ssl_model, ssl_out_dim):
        super(MosPredictor, self).__init__()
        self.ssl_model = ssl_model
        self.ssl_features = ssl_out_dim

    def forward(self, wav):
        wav = wav.squeeze(1)  ## [batches, audio_len]
        res = self.ssl_model(wav, mask=False, features_only=True)
        x = res['x']

        return x


class PoolAtt(torch.nn.Module):
    '''
    PoolAtt: Attention-Pooling module.
    '''
    def __init__(self, d_input, output_size):
        super().__init__()

        self.linear1 = nn.Linear(d_input, 1)
        self.linear2 = nn.Linear(d_input, output_size)

    def forward(self, x, n_wins):

        att = self.linear1(x) # B X T X C

        att = att.transpose(2,1) # B X 1 X T
        mask = torch.arange(att.shape[2])[None, :] < n_wins[:, None].to('cpu').to(torch.long)
        att[~mask.unsqueeze(1)] = float("-Inf")
        att = F.softmax(att, dim=2)
        x = torch.bmm(att, x)
        x = x.squeeze(1)
        x = self.linear2(x)

        return x
