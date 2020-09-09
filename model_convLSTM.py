import math
import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
from torch.autograd import Variable
from torchvision.models import resnet50, resnet101, resnet152
from manopth.manolayer import ManoLayer
from utils import batch_rodrigues
from lstms import ConvLSTM

""" Code from https://github.com/eriklindernoren/Action-Recognition """


def getBackbone(backbone, pretrain=False):
    if backbone == 'resnet50':
        model = resnet50(pretrained=pretrain)
    elif backbone == 'resnet101':
        model = resnet101(pretrained=pretrain)
    elif backbone == 'resnet152':
        model = resnet152(pretrained=pretrain)
    else:
        raise Exception('Undefined configuration...')
    return model


class Encoder(nn.Module):
    def __init__(self, latent_dim, backbone, pretrain=False):
        super(Encoder, self).__init__()
        resnet = getBackbone(backbone, pretrain)
        self.feature_extractor = nn.Sequential(*list(resnet.children())[:-2])
        self.dim_reduction = nn.Sequential(
            nn.Conv2d(resnet.fc.in_features, latent_dim, kernel_size=1),
            nn.BatchNorm2d(latent_dim, momentum=0.01)
        )

    def forward(self, x):
        batch_size, seq_size, c, h, w = x.size()        # [batch_size, seq_size,   3, 224, 224]
        x = x.view(batch_size * seq_size, c, h, w)      # [batch_size*seq_size,    3, 224, 224]
        x = self.feature_extractor(x)                   # [batch_size*seq_size, 2048,  h',  w']
        x = self.dim_reduction(x)                       # [batch_size*seq_size,  512,  h',  w']
        _, c, h, w = x.size()
        x = x.view(batch_size, seq_size, c, h, w)
        return x


class EncoderConvLSTM(nn.Module):
    def __init__(self, args, x, latent_dim=512, hidden_dim=1024, lstm_layers=1, attention=False, n_classes=26, backbone='resnet50', pretrain=False):
        super(EncoderConvLSTM, self).__init__()
        self.camera_s = {'mean': 753.188477, 'std': 120.778145}
        self.camera_u = {'mean': 94.337875, 'std': 13.138042}
        self.camera_v = {'mean': 92.918877, 'std': 25.438021}
        self.encoder = Encoder(latent_dim, backbone, pretrain)
        self.conv_lstm = ConvLSTM(input_size=(7, 7), input_dim=latent_dim, hidden_dim=hidden_dim, kernel_size=(3, 3), num_layers=lstm_layers, batch_first=True, bias=True, return_all_layers=False)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # self.output_layers = nn.Sequential(
        #     nn.Linear(hidden_dim, hidden_dim),
        #     nn.BatchNorm1d(hidden_dim, momentum=0.01),
        #     nn.ReLU(),
        #     nn.Linear(hidden_dim, n_classes)
        # )
        # 20191019 Dropout & Simple output layers for generalization
        self.output_layers = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, n_classes, bias=False)
        )
        self.mano_layerL = ManoLayer(mano_root=args.root_mano, use_pca=True, ncomps=10, flat_hand_mean=False, side='left')  # scaled to milimeters
        self.mano_layerR = ManoLayer(mano_root=args.root_mano, use_pca=True, ncomps=10, flat_hand_mean=False, side='right')  # scaled to milimeters
        # self.attention = attention
        # self.attention_layer = nn.Linear(2 * hidden_dim if bidirectional else hidden_dim, 1)  # Linear(in_features=2048, out_features=1, bias=True)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data = torch.nn.init.kaiming_normal_(m.weight.data)

    def forward(self, x):
        batch_size, seq_size, _, _, _ = x.size()
        x = self.encoder(x)                             # [batch_size, seq_size, latent_dim, h', c']
        x, hx = self.conv_lstm(x, None)                 # [batch_size, seq_size, hidden_dim, h', c'], [[seq_size, hidden_dim, h', c'], [seq_size, hidden_dim, h', c']]
        _, _, c, h, w = x.size()
        x = x.view(batch_size*seq_size, c, h, w)        # [batch_size*seq_size, hidden_dim, h', c']
        x = self.avg_pool(x)                            # [batch_size*seq_size, hidden_dim,  1,  1]
        x = x.view(batch_size*seq_size, -1)             # [batch_size*seq_size, hidden_dim]
        x = self.output_layers(x)                       # [batch_size*seq_size, n_classes]
        # Get hand joints
        x2d, x3d, camera_param, theta, beta = self.getHandJoints(x, batch_size, seq_size)
        return x2d, x3d, camera_param, theta, beta

    def getHandJoints(self, xs, b_size, seq_size):
        # Initialize Camera Parameters
        scale = xs[:, 0]  # Scale       1
        trans = xs[:, 1:3]  # Translate   2
        rotation = xs[:, 3:6]  # Rotation    3
        theta = xs[:, 6:16]  # Theta      10
        beta = xs[:, 16:]  # Beta       10
        scale = scale * torch.tensor([self.camera_s['std']]).cuda() + torch.tensor([self.camera_s['mean']]).cuda()
        trans = trans * torch.tensor([self.camera_u['std'], self.camera_v['std']]).unsqueeze(0).cuda() + torch.tensor([self.camera_u['mean'], self.camera_v['mean']]).unsqueeze(0).cuda()

        batch_size = theta.size(0)
        theta_ = torch.cat((torch.tensor([math.pi, 0., 0.]).unsqueeze(0).repeat(batch_size, 1).float().cuda(), theta), 1)  # [pi, 0, 0, theta]

        verts_3d_pred, joint_3d_pred = self.mano_layerR(theta_, beta)

        R = batch_rodrigues(rotation).view(batch_size, 3, 3)
        joint_3d_pred = torch.transpose(torch.matmul(R, torch.transpose(joint_3d_pred, 1, 2)), 1, 2) / 1000  # [mm] >> [m]
        verts_3d_pred = torch.transpose(torch.matmul(R, torch.transpose(verts_3d_pred, 1, 2)), 1, 2) / 1000  # [mm] >> [m]

        joint_2d_pred = joint_3d_pred[:, :, :2] * scale.unsqueeze(1).unsqueeze(2) + trans.unsqueeze(1)  # [pixel]
        verts_2d_pred = verts_3d_pred[:, :, :2] * scale.unsqueeze(1).unsqueeze(2) + trans.unsqueeze(1)  # [pixel]

        x3d = torch.cat((joint_3d_pred, verts_3d_pred), dim=1)
        x2d = torch.cat((joint_2d_pred, verts_2d_pred), dim=1).view(batch_size, -1)

        x2d = x2d.view(b_size, seq_size, -1)
        x3d = x3d.view(b_size, seq_size, -1)
        xs = xs.view(b_size, seq_size, -1)
        theta = theta.view(b_size, seq_size, -1)
        beta = beta.view(b_size, seq_size, -1)

        return x2d, x3d, xs, theta, beta


class ConvClassifier(nn.Module):
    def __init__(self, num_classes, latent_dim, backbone='resnet50', pretrain=False):
        super(ConvClassifier, self).__init__()
        resnet = getBackbone(backbone, pretrain)
        self.feature_extractor = nn.Sequential(*list(resnet.children())[:-1])
        self.final = nn.Sequential(
            nn.Linear(resnet.fc.in_features, latent_dim),
            nn.BatchNorm1d(latent_dim, momentum=0.01),
            nn.Linear(latent_dim, num_classes),
            nn.Softmax(dim=-1),
        )

    def forward(self, x):
        batch_size, seq_length, c, h, w = x.shape
        x = x.view(batch_size * seq_length, c, h, w)
        x = self.feature_extractor(x)
        x = x.view(batch_size * seq_length, -1)
        x = self.final(x)
        x = x.view(batch_size, seq_length, -1)
        return x
