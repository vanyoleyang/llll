from .child_models.Resnet import Encoder as ResnetEncoder
from .modeling_lib.mano.manolayer import ManoLayer

import torch
import torch.nn as nn
from .modeling_lib.smpl.smpl_torch_batch import SMPLModel

class Hand_pose_est_model(nn.Module):
    def __init__(self, args, root_mano, hidden_dim=512, n_classes = 26, child_model='resnet50'):
        super(Hand_pose_est_model, self).__init__()
        self.args = args
        self.device = args.device_ch
        if 'resnet' in child_model :
            self.model = ResnetEncoder(latent_dim=512, backbone=child_model)
        self.output_layers = nn.Sequential(
            nn.Linear(self.model.output_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU())
        self.pose_est = nn.Sequential(
            nn.Linear(hidden_dim, n_classes))
        self.disc_est = nn.Sequential(
            nn.Linear(hidden_dim, 1), nn.Sigmoid())

        self.theta_sp = 10  # rotataion 3 + pose_param 7
        self.beta_sp = 10
        self.trans_sp = 2
        self.mano_layer = ManoLayer(mano_root=root_mano, use_pca=True, ncomps=self.theta_sp-3,
                                    flat_hand_mean=False, side='right')
        for param in self.model.parameters():
            param.requires_grad = True
        for param in self.output_layers.parameters():
            param.requires_grad = True
        for param in self.mano_layer.parameters():
            param.requires_grad = False

        self.sc_mean, self.sc_std, self.trans_0_std, self.trans_0_mean, self.trans_1_std, self.trans_1_mean \
            = 813., 151., 54., 45, 54., 120.

    def forward(self, x):
        x = self.model(x.float())
        xx = self.output_layers(x)
        x = self.pose_est(xx)
        y = self.disc_est(xx)
        # return self.getHandJoints(x)
        pose, shape, trans, scale = self.get_pose_shape_trans_scale(x)
        verts_3d, joints_3d = self.get_3d_verts_and_joints(pose, shape)
        verts2d, joints2d = self.get_2d_verts_and_joints(verts_3d, joints_3d, scale, trans)
        return (verts_3d, joints_3d, verts2d[:, :, :2], joints2d[:, :, :2], pose, shape, trans, scale, y)

    def get_pose_shape_trans_scale(self, input_):
        pose = input_[:, :self.theta_sp]
        shape = input_[:, self.theta_sp:self.theta_sp + self.beta_sp]
        trans = input_[:, self.theta_sp + self.beta_sp:self.theta_sp + self.beta_sp + self.trans_sp]
        scale = input_[:, -1]
        if not self.args.body:
            trans[:, 0] = trans[:, 0] * self.trans_0_std + self.trans_0_mean
            trans[:, 1] = trans[:, 1] * self.trans_1_std + self.trans_1_mean
            scale = scale * self.sc_std + self.sc_mean
        return pose, shape, trans, scale

    def get_2d_verts_and_joints(self, verts_3d, joints_3d, scale, trans):
        trans = torch.cat([trans, torch.zeros(trans.size(0), 1).to(self.device)], dim=1)
        joints_2d = joints_3d * scale.unsqueeze(1).unsqueeze(2) + trans.unsqueeze(1)
        verts_2d = verts_3d * scale.unsqueeze(1).unsqueeze(2) + trans.unsqueeze(1)
        return verts_2d, joints_2d

    def get_3d_verts_and_joints(self, pose, shape):
        verts, joints = self.mano_layer(pose, shape)
        return verts, joints




class Body_pose_est_model(nn.Module):
    def __init__(self, hidden_dim=512, n_classes=72, child_model='resnet50'):
        super(Body_pose_est_model, self).__init__()
        if child_model =='resnet50':
            self.model = Resnet50Encoder(backbone='resnet50')
        self.output_layers = nn.Sequential(
            nn.Linear(self.model.output_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim, momentum=0.01),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_classes)
        )
        self.smpl_layer = SMPLModel(device='cuda:0')


