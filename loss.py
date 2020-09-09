import torch
import torch.nn as nn
import numpy as np
from utils import displayImage, displayMask




class FHADLoss(nn.Module):
    def __init__(self, args):
        super(FHADLoss, self).__init__()
        # Initialize Parameters
        self.args = args
        self.alpha_2d = 2
        self.alpha_3d = 10 #100
        self.alpha_mask = 10 # 100
        self.alpha_reg = 0#10
        self.alpha_beta = 0#10000
        self.alpha_camera = 1

        self.img_size = 224

    def forward(self, epoch, mask, predictions, targets, train=True):
        x2d_pred, x3d_pred, _, theta, beta = predictions
        joint_2d_targ, joint_3d_targ, _, _, _ = targets
        # Initialize Variables
        batch_size, seq_size, _ = x2d_pred.size()
        joint_3d_pred = torch.stack((x3d_pred[:, :, :63:3], x3d_pred[:, :, 1:63:3], x3d_pred[:, :, 2:63:3]),
                                    dim=3)  # out2[:, :21, :]
        joint_3d_pred, pred_min, pred_max = self.normalize_joints_scale(joint_3d_pred)
        joint_3d_targ, targ_min, targ_max = self.normalize_joints_scale(joint_3d_targ)
        _, _, maxp = self.normalize_joints_scale(joint_3d_pred)
        _, _, maxt = self.normalize_joints_scale(joint_3d_targ)
        joint_3d_pred = self.center_joints_scale(joint_3d_pred, maxp)
        joint_3d_targ = self.center_joints_scale(joint_3d_targ, maxt)
        joint_3d_pred, joint_3d_targ = joint_3d_pred[:, -1, :, :], joint_3d_targ[:, -1, :, :]

        joint_2d_pred = torch.stack((x2d_pred[:, -1, :42:2], x2d_pred[:, -1, 1:42:2]), dim=2)  # x_hat
        joint_2d_targ = joint_2d_targ[:, -1, :, :]
        loss_2d = torch.abs((joint_2d_pred.view(batch_size, -1) / self.img_size
                             - joint_2d_targ.view(batch_size, -1) / self.img_size)).sum(1).mean()
        loss_2d = self.alpha_2d * loss_2d
        diff_2d = joint_2d_pred.view(batch_size, -1, 2) - joint_2d_targ.view(batch_size, -1,2)

        diff_3d = joint_3d_pred.view(batch_size, -1, 3) - joint_3d_targ.view(batch_size, -1,3)

        loss_3d = torch.pow(diff_3d.view(batch_size, -1), 2).sum(1).mean()
        diff_3d = diff_3d * (pred_max - pred_min).repeat(1, 1, 21, 1)[:, -1, :, :].view(batch_size, -1, 3)
        loss_3d = self.alpha_3d * loss_3d
        theta_prev = torch.cat((theta[:, 0, :].unsqueeze(1), theta[:, :-1, :]), 1)
        beta_prev = torch.cat((beta[:, 0, :].unsqueeze(1), beta[:, :-1, :]), 1)
        pose_temp_loss = torch.pow(theta_prev.view(batch_size * seq_size, -1) - theta.view(batch_size * seq_size, -1), 2).sum(1).mean()
        shape_temp_loss = torch.pow(beta_prev.view(batch_size * seq_size, -1) - beta.view(batch_size * seq_size, -1), 2).sum(1).mean()
        loss_temp = 0.0005 * ( 0.1 * pose_temp_loss + 1. * shape_temp_loss)
        loss_mask = torch.zeros(1).to(self.args.device)
        loss_reg = torch.zeros(1).to(self.args.device)
        loss_camera = torch.zeros(1).to(self.args.device)

        loss = loss_2d + loss_3d + loss_mask + loss_reg + loss_camera + loss_temp

        # Initialize Average Distance Storage
        avg_distance_2d = list()
        avg_distance_3d = list()
        for _ in range(self.args.n_kps):
            avg_distance_2d.append(None)
            avg_distance_3d.append(None)
        # Calculate euclidean distance

        euclidean_dist_2d = np.sqrt(np.sum(np.square(diff_2d.detach().cpu().numpy()), axis=2))
        euclidean_dist_3d = np.sqrt(np.sum(np.square(diff_3d.detach().cpu().numpy()), axis=2))
        for i in range(self.args.n_kps):
            avg_distance_2d[i] = euclidean_dist_2d[:, i]
            avg_distance_3d[i] = euclidean_dist_3d[:, i]
        return loss, [loss_2d.item(), loss_3d.item(), loss_temp.item(), loss_reg.item(), loss_camera.item(), avg_distance_2d, avg_distance_3d]


    def normalize_joints_scale(self, hand_joints):
        min_joints, _ = torch.min(hand_joints, dim=2, keepdim=True)
        max_joints, _ = torch.max(hand_joints, dim=2, keepdim=True)
        hand_joints[:, :, :, 0] = (hand_joints[:, :, :, 0] - min_joints[:, :, :, 0]) / (max_joints[:, :, :, 0] - min_joints[:, :, :, 0])
        hand_joints[:, :, :, 1] = (hand_joints[:, :, :, 1] - min_joints[:, :, :, 1]) / (max_joints[:, :, :, 0] - min_joints[:, :, :, 0])
        hand_joints[:, :, :, 2] = (hand_joints[:, :, :, 2] - min_joints[:, :, :, 2]) / (max_joints[:, :, :, 0] - min_joints[:, :, :, 0])
        return hand_joints, min_joints, max_joints


    def center_joints_scale(self, hand_joints, max_joints):
        hand_joints[:, :, :, 0] = hand_joints[:, :, :, 0]  - max_joints[:, :, :, 0]
        hand_joints[:, :, :, 1] =  hand_joints[:, :, :, 1]   -max_joints[:, :, :, 1]
        hand_joints[:, :, :, 2] =  hand_joints[:, :, :, 2]  - max_joints[:, :, :, 2]
        return hand_joints



class Hand3DLoss_wKLD(nn.Module):
    def __init__(self, args, pretrain=False):
        super(Hand3DLoss_wKLD, self).__init__()
        # Initialize Parameters
        self.args = args

        self.alpha_2d = 5
        self.alpha_3d = 10 #100
        self.alpha_mask = 10 # 100
        self.alpha_reg = 0#10
        self.alpha_beta = 0#10000
        self.alpha_camera = 1
        self.alpha_kld = 0.00001
        self.n_meshes = 778
        self.img_size = 224

    def forward(self, epoch, mask, predictions, targets):
        # Initialize predictions
        x2d_pred, x3d_pred, camera_param_pred, theta, beta, mu, logvar = predictions
        # Initialize targets
        joint_2d_target, joint_3d_target, verts_3d_target, camera_param_target, dataset_type = targets
        # Initialize Variables
        batch_size, seq_size, _ = x2d_pred.size()
        # Get Vectors
        joint_2d_pred = torch.stack((x2d_pred[:, :, :42:2], x2d_pred[:, :, 1:42:2]), dim=3)  # x_hat
        y_hat = x2d_pred[:, :, 42:].view(batch_size, seq_size, 778, 2)
        joint_3d_pred = torch.stack((x3d_pred[:, :, :63:3], x3d_pred[:, :, 1:63:3], x3d_pred[:, :, 2:63:3]), dim=3)  # out2[:, :21, :]
        # KLD loss
        loss_kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        loss_kld = self.alpha_kld * loss_kld
        # Calculate the Losses - 2D joint re-projection loss
        loss_2d = torch.abs((joint_2d_pred.view(batch_size*seq_size, -1) / self.img_size - joint_2d_target.view(batch_size*seq_size, -1) / self.img_size)).sum(1).mean()
        loss_2d = self.alpha_2d * loss_2d
        # Calculate the Losses - 3D joint loss (Only the STEREO dataset)
        # print(joint_3d_pred[0, 0, 0, :], joint_3d_target[0, 0, 0, :])

        joint_3d_pred, pred_min, pred_max = self.normalize_joints_scale(joint_3d_pred)
        joint_3d_target, targ_min, targ_max = self.normalize_joints_scale(joint_3d_target)
        _, _, maxp = self.normalize_joints_scale(joint_3d_pred)
        _, _, maxt = self.normalize_joints_scale(joint_3d_target)
        joint_3d_pred = self.center_joints_scale(joint_3d_pred, maxp)
        joint_3d_target = self.center_joints_scale(joint_3d_target,maxt )
        # self.draw_3d_mano_pose(joint_3d_pred[0, 0, :, :], joint_3d_target[0, 0, :, :])

        diff_3d = joint_3d_pred.view(batch_size * seq_size, -1, 3) - joint_3d_target.view(batch_size * seq_size, -1, 3)
        loss_3d = torch.pow(diff_3d.view(batch_size * seq_size, -1), 2).sum(1).mean()

        diff_3d = diff_3d * (pred_max - pred_min).repeat(1, 1, 21, 1).view(batch_size * seq_size, -1, 3)
        loss_3d = self.alpha_3d * loss_3d


        theta_prev = torch.cat((theta[:, 0, :].unsqueeze(1), theta[:, :-1, :]), 1)
        beta_prev = torch.cat((beta[:, 0, :].unsqueeze(1), beta[:, :-1, :]), 1)
        pose_temp_loss = torch.pow(theta_prev.view(batch_size * seq_size, -1) - theta.view(batch_size * seq_size, -1),
                                   2).sum(1).mean()
        shape_temp_loss = torch.pow(beta_prev.view(batch_size * seq_size, -1) - beta.view(batch_size * seq_size, -1),
                                    2).sum(1).mean()
        # print('theta_temp loss', pose_temp_loss, 'beta_temp_loss', shape_temp_loss)
        if 'ConvLSTM' in self.args.model_name :
            loss_temp =  ( 0.05*pose_temp_loss + 100. * shape_temp_loss)
            # loss_temp *= 0.
        else : loss_temp = torch.zeros(1).cuda()
        # Calculate the Losses - Hand mask loss
        loss_mask = self.getHandMask(y_hat, mask)
        loss_mask = self.alpha_mask * loss_mask
        # Calculate the Losses - Regularization loss
        loss_reg = torch.zeros(1).cuda()
        # Calculate the Losses - Camera Parameter Loss
        if camera_param_target.sum().abs().item() > 0:
            if dataset_type[0] == 7 :
                loss_camera = torch.nn.functional.mse_loss(
                    camera_param_pred[:, :, 16:26].view(batch_size * seq_size, -1),
                    camera_param_target[:, :, 16:26].view(batch_size * seq_size, -1))
            else :
                loss_camera_scale = torch.nn.functional.mse_loss(camera_param_pred[:, :, 0:1].view(batch_size*seq_size, -1), camera_param_target[:, :, 0:1].view(batch_size*seq_size, -1))
                loss_camera_trans = torch.nn.functional.mse_loss(camera_param_pred[:, :, 1:3].view(batch_size*seq_size, -1), camera_param_target[:, :, 1:3].view(batch_size*seq_size, -1))
                loss_camera_rot = torch.nn.functional.mse_loss(camera_param_pred[:, :, 3:6].view(batch_size*seq_size, -1), camera_param_target[:, :, 3:6].view(batch_size*seq_size, -1))
                loss_camera_theta = torch.nn.functional.mse_loss(camera_param_pred[:, :, 6:16].view(batch_size*seq_size, -1), camera_param_target[:, :, 6:16].view(batch_size*seq_size, -1))
                loss_camera_beta = torch.nn.functional.mse_loss(camera_param_pred[:, :, 16:26].view(batch_size*seq_size, -1), camera_param_target[:, :, 16:26].view(batch_size*seq_size, -1))
                loss_camera = loss_camera_scale + loss_camera_trans + loss_camera_rot + loss_camera_theta + loss_camera_beta
                theta_dif = np.sum(np.square((camera_param_pred[:, :, 6:16].view(batch_size, seq_size, -1) -
                             camera_param_target[:, :, 6:16].view(batch_size, seq_size, -1) ).detach().cpu().numpy()), axis=0)
                beta_dif = np.sum(np.square((camera_param_pred[:, :, 16:26].view(batch_size, seq_size, -1) -
                             camera_param_target[:, :, 16:26].view(batch_size, seq_size, -1) ).detach().cpu().numpy()), axis=0)
                print('theta ', theta_dif,'\n', 'beta ', beta_dif)
            loss_camera = self.alpha_camera * loss_camera
        else:
            loss_camera = torch.zeros(1).cuda()
        # Weighted sum
        loss = loss_2d + loss_3d + loss_mask + loss_reg + loss_camera + loss_temp + loss_kld

        # Initialize Average Distance Storage
        avg_distance_2d = list()
        avg_distance_3d = list()
        for _ in range(self.args.n_kps):
            avg_distance_2d.append(None)
            avg_distance_3d.append(None)
        # Calculate euclidean distance
        diff_2d = joint_2d_pred.view(batch_size*seq_size, -1, 2) - joint_2d_target.view(batch_size*seq_size, -1, 2)

        euclidean_dist_2d = np.sqrt(np.sum(np.square(diff_2d.detach().cpu().numpy()), axis=2))
        euclidean_dist_3d = np.sqrt(np.sum(np.square(diff_3d.detach().cpu().numpy()), axis=2))
        for i in range(self.args.n_kps):
            avg_distance_2d[i] = euclidean_dist_2d[:, i]
            avg_distance_3d[i] = euclidean_dist_3d[:, i]
        return loss, [loss_2d.item(), loss_3d.item(), loss_mask.item(), loss_kld.item() , loss_camera.item(), avg_distance_2d, avg_distance_3d]

    def getHandMask(self, y_hat, mask):
        batch_size, seq_size, _, h, w = mask.size()
        loss_mask = torch.ones(batch_size, seq_size, 1).cuda()
        y_hat = y_hat.round().long()
        y_hat[:, :, :, 0] = torch.where(y_hat[:, :, :, 0] >= w, torch.tensor(w-1, dtype=torch.long).cuda(), y_hat[:, :, :, 0])
        y_hat[:, :, :, 1] = torch.where(y_hat[:, :, :, 1] >= h, torch.tensor(h-1, dtype=torch.long).cuda(), y_hat[:, :, :, 1])
        y_hat[:, :, :, 0] = torch.where(y_hat[:, :, :, 0] < 0, torch.tensor(0, dtype=torch.long).cuda(), y_hat[:, :, :, 0])
        y_hat[:, :, :, 1] = torch.where(y_hat[:, :, :, 1] < 0, torch.tensor(0, dtype=torch.long).cuda(), y_hat[:, :, :, 1])
        for i_batch in range(batch_size):
            for i_seq in range(seq_size):
                loss_mask[i_batch, i_seq] = loss_mask[i_batch, i_seq] - mask[i_batch, i_seq, 0, y_hat[i_batch, i_seq, :, 1], y_hat[i_batch, i_seq, :, 0]].sum()/self.n_meshes
        return loss_mask.mean()

    def normalize_joints_scale(self, hand_joints):
        min_joints, _ = torch.min(hand_joints, dim=2, keepdim=True)
        max_joints, _ = torch.max(hand_joints, dim=2, keepdim=True)
        hand_joints[:, :, :, 0] = (hand_joints[:, :, :, 0] - min_joints[:, :, :, 0]) / (max_joints[:, :, :, 0] - min_joints[:, :, :, 0])
        hand_joints[:, :, :, 1] = (hand_joints[:, :, :, 1] - min_joints[:, :, :, 1]) / (max_joints[:, :, :, 0] - min_joints[:, :, :, 0])
        hand_joints[:, :, :, 2] = (hand_joints[:, :, :, 2] - min_joints[:, :, :, 2]) / (max_joints[:, :, :, 0] - min_joints[:, :, :, 0])
        return hand_joints, min_joints, max_joints


    def center_joints_scale(self, hand_joints, max_joints):
        hand_joints[:, :, :, 0] = hand_joints[:, :, :, 0]  - max_joints[:, :, :, 0]
        hand_joints[:, :, :, 1] =  hand_joints[:, :, :, 1]   -max_joints[:, :, :, 1]
        hand_joints[:, :, :, 2] =  hand_joints[:, :, :, 2]  - max_joints[:, :, :, 2]
        return hand_joints


class Hand3DLoss(nn.Module):
    def __init__(self, args, pretrain=False):
        super(Hand3DLoss, self).__init__()
        # Initialize Parameters
        self.args = args
        self.pretrain = pretrain
        if self.pretrain:
            self.alpha_2d = 0
            self.alpha_3d = 0
            self.alpha_mask = 0
            self.alpha_reg = 0
            self.alpha_beta = 0
            self.alpha_camera = 1
        else:
            self.alpha_2d = 5
            self.alpha_3d = 100 #100
            self.alpha_mask = 10 # 100
            self.alpha_reg = 0#10
            self.alpha_beta = 0#10000
            self.alpha_camera = 1
        self.n_meshes = 778
        self.img_size = 224

    def getRampUpScale(self, epoch):
        if self.pretrain:
            return torch.ones(1).cuda()
        else:
            return torch.ones(1).cuda()
            # return torch.FloatTensor([(epoch+1) / self.args.max_epochs_ramp_up]).cuda()

    def forward(self, epoch, mask, predictions, targets):
        # Initialize RampUp Scale
        rampup_scale = self.getRampUpScale(epoch)
        # Initialize predictions
        x2d_pred, x3d_pred, camera_param_pred, theta, beta = predictions
        # Initialize targets
        joint_2d_target, joint_3d_target, verts_3d_target, camera_param_target, dataset_type = targets
        # Initialize Variables
        batch_size, seq_size, _ = x2d_pred.size()
        # Get Vectors
        joint_2d_pred = torch.stack((x2d_pred[:, :, :42:2], x2d_pred[:, :, 1:42:2]), dim=3)  # x_hat
        y_hat = x2d_pred[:, :, 42:].view(batch_size, seq_size, 778, 2)
        joint_3d_pred = torch.stack((x3d_pred[:, :, :63:3], x3d_pred[:, :, 1:63:3], x3d_pred[:, :, 2:63:3]), dim=3)  # out2[:, :21, :]
        # Calculate the Losses - 2D joint re-projection loss
        loss_2d = torch.abs((joint_2d_pred.view(batch_size*seq_size, -1) / self.img_size - joint_2d_target.view(batch_size*seq_size, -1) / self.img_size)).sum(1).mean()
        loss_2d = rampup_scale * self.alpha_2d * loss_2d
        # Calculate the Losses - 3D joint loss (Only the STEREO dataset)
        # print(joint_3d_pred[0, 0, 0, :], joint_3d_target[0, 0, 0, :])

        # joint_3d_pred, pred_min, pred_max = self.normalize_joints_scale(joint_3d_pred)
        # joint_3d_target, targ_min, targ_max = self.normalize_joints_scale(joint_3d_target)
        # _, _, maxp = self.normalize_joints_scale(joint_3d_pred)
        # _, _, maxt = self.normalize_joints_scale(joint_3d_target)
        # joint_3d_pred = self.center_joints_scale(joint_3d_pred, maxp)
        # joint_3d_target = self.center_joints_scale(joint_3d_target,maxt )
        # self.draw_3d_mano_pose(joint_3d_pred[0, 0, :, :], joint_3d_target[0, 0, :, :])

        diff_3d = joint_3d_pred.view(batch_size * seq_size, -1, 3) - joint_3d_target.view(batch_size * seq_size, -1, 3)
        loss_3d = torch.pow(diff_3d.view(batch_size * seq_size, -1), 2).sum(1).mean()

        #diff_3d = diff_3d * (pred_max - pred_min).repeat(1, 1, 21, 1).view(batch_size * seq_size, -1, 3)
        loss_3d = rampup_scale * self.alpha_3d * loss_3d


        theta_prev = torch.cat((theta[:, 0, :].unsqueeze(1), theta[:, :-1, :]), 1)
        beta_prev = torch.cat((beta[:, 0, :].unsqueeze(1), beta[:, :-1, :]), 1)
        pose_temp_loss = torch.pow(theta_prev.view(batch_size * seq_size, -1) - theta.view(batch_size * seq_size, -1),
                                   2).sum(1).mean()
        shape_temp_loss = torch.pow(beta_prev.view(batch_size * seq_size, -1) - beta.view(batch_size * seq_size, -1),
                                    2).sum(1).mean()
        # print('theta_temp loss', pose_temp_loss, 'beta_temp_loss', shape_temp_loss)
        if 'ConvLSTM' in self.args.model_name :
            loss_temp =   0.*( 0.05 * pose_temp_loss + 1. * shape_temp_loss)
            # loss_temp *= 0.
        else : loss_temp = torch.zeros(1).cuda()
        # Calculate the Losses - Hand mask loss
        loss_mask = self.getHandMask(y_hat, mask)
        loss_mask = rampup_scale * self.alpha_mask * loss_mask
        # Calculate the Losses - Regularization loss
        loss_reg = torch.zeros(1).cuda()
        # Calculate the Losses - Camera Parameter Loss
        if camera_param_target.sum().abs().item() > 0:
            if dataset_type[0] == 7 :
                loss_camera = torch.nn.functional.mse_loss(
                    camera_param_pred[:, :, 16:26].view(batch_size * seq_size, -1),
                    camera_param_target[:, :, 16:26].view(batch_size * seq_size, -1))
            else :
                loss_camera_scale = torch.nn.functional.mse_loss(camera_param_pred[:, :, 0:1].view(batch_size*seq_size, -1), camera_param_target[:, :, 0:1].view(batch_size*seq_size, -1))
                loss_camera_trans = torch.nn.functional.mse_loss(camera_param_pred[:, :, 1:3].view(batch_size*seq_size, -1), camera_param_target[:, :, 1:3].view(batch_size*seq_size, -1))
                loss_camera_rot = torch.nn.functional.mse_loss(camera_param_pred[:, :, 3:6].view(batch_size*seq_size, -1), camera_param_target[:, :, 3:6].view(batch_size*seq_size, -1))
                loss_camera_theta = torch.nn.functional.mse_loss(camera_param_pred[:, :, 6:16].view(batch_size*seq_size, -1), camera_param_target[:, :, 6:16].view(batch_size*seq_size, -1))
                loss_camera_beta = torch.nn.functional.mse_loss(camera_param_pred[:, :, 16:26].view(batch_size*seq_size, -1), camera_param_target[:, :, 16:26].view(batch_size*seq_size, -1))

                loss_camera = loss_camera_scale + loss_camera_trans + loss_camera_rot + loss_camera_theta + loss_camera_beta

                theta_dif = np.sum(np.square((camera_param_pred[:, :, 6:16].view(batch_size, seq_size, -1) -
                             camera_param_target[:, :, 6:16].view(batch_size, seq_size, -1) ).detach().cpu().numpy()), axis=0)
                beta_dif = np.sum(np.square((camera_param_pred[:, :, 16:26].view(batch_size, seq_size, -1) -
                             camera_param_target[:, :, 16:26].view(batch_size, seq_size, -1) ).detach().cpu().numpy()), axis=0)
                print(theta_dif, beta_dif)


            loss_camera = self.alpha_camera * loss_camera
        else:
            loss_camera = torch.zeros(1).cuda()
        # Weighted sum
        loss = loss_2d + loss_3d + loss_mask + loss_reg + loss_camera + loss_temp

        # Initialize Average Distance Storage
        avg_distance_2d = list()
        avg_distance_3d = list()
        for _ in range(self.args.n_kps):
            avg_distance_2d.append(None)
            avg_distance_3d.append(None)
        # Calculate euclidean distance
        diff_2d = joint_2d_pred.view(batch_size*seq_size, -1, 2) - joint_2d_target.view(batch_size*seq_size, -1, 2)

        euclidean_dist_2d = np.sqrt(np.sum(np.square(diff_2d.detach().cpu().numpy()), axis=2))
        euclidean_dist_3d = np.sqrt(np.sum(np.square(diff_3d.detach().cpu().numpy()), axis=2))
        for i in range(self.args.n_kps):
            avg_distance_2d[i] = euclidean_dist_2d[:, i]
            avg_distance_3d[i] = euclidean_dist_3d[:, i]
        return loss, [loss_2d.item(), loss_3d.item(), loss_mask.item(), loss_reg.item(), loss_camera.item(), avg_distance_2d, avg_distance_3d]

    def getHandMask(self, y_hat, mask):
        batch_size, seq_size, _, h, w = mask.size()
        loss_mask = torch.ones(batch_size, seq_size, 1).cuda()
        y_hat = y_hat.round().long()
        y_hat[:, :, :, 0] = torch.where(y_hat[:, :, :, 0] >= w, torch.tensor(w-1, dtype=torch.long).cuda(), y_hat[:, :, :, 0])
        y_hat[:, :, :, 1] = torch.where(y_hat[:, :, :, 1] >= h, torch.tensor(h-1, dtype=torch.long).cuda(), y_hat[:, :, :, 1])
        y_hat[:, :, :, 0] = torch.where(y_hat[:, :, :, 0] < 0, torch.tensor(0, dtype=torch.long).cuda(), y_hat[:, :, :, 0])
        y_hat[:, :, :, 1] = torch.where(y_hat[:, :, :, 1] < 0, torch.tensor(0, dtype=torch.long).cuda(), y_hat[:, :, :, 1])
        for i_batch in range(batch_size):
            for i_seq in range(seq_size):
                loss_mask[i_batch, i_seq] = loss_mask[i_batch, i_seq] - mask[i_batch, i_seq, 0, y_hat[i_batch, i_seq, :, 1], y_hat[i_batch, i_seq, :, 0]].sum()/self.n_meshes
        return loss_mask.mean()

    def normalize_joints_scale(self, hand_joints):
        min_joints, _ = torch.min(hand_joints, dim=2, keepdim=True)
        max_joints, _ = torch.max(hand_joints, dim=2, keepdim=True)
        hand_joints[:, :, :, 0] = (hand_joints[:, :, :, 0] - min_joints[:, :, :, 0]) / (max_joints[:, :, :, 0] - min_joints[:, :, :, 0])
        hand_joints[:, :, :, 1] = (hand_joints[:, :, :, 1] - min_joints[:, :, :, 1]) / (max_joints[:, :, :, 0] - min_joints[:, :, :, 0])
        hand_joints[:, :, :, 2] = (hand_joints[:, :, :, 2] - min_joints[:, :, :, 2]) / (max_joints[:, :, :, 0] - min_joints[:, :, :, 0])
        return hand_joints, min_joints, max_joints


    def center_joints_scale(self, hand_joints, max_joints):
        hand_joints[:, :, :, 0] = hand_joints[:, :, :, 0]  - max_joints[:, :, :, 0]
        hand_joints[:, :, :, 1] =  hand_joints[:, :, :, 1]   -max_joints[:, :, :, 1]
        hand_joints[:, :, :, 2] =  hand_joints[:, :, :, 2]  - max_joints[:, :, :, 2]
        return hand_joints


    def draw_3d_mano_pose(self, pose_3d, pose_3d2, color='black', color2 ='red'):
        pose_3d = pose_3d.reshape(21, 3)
        pose_3d2 = pose_3d2.reshape(21, 3)
        # print(pose_3d[0][0], pose_3d[0][1], pose_3d[0][2])
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        b = color # or 'red'
        ax.plot([pose_3d2[0][0], pose_3d2[1][0]], [pose_3d2[0][1], pose_3d2[1][1]], zs=[pose_3d2[0][2], pose_3d2[1][2]],
                linewidth=3, color=color2)
        ax.plot([pose_3d2[0][0], pose_3d2[5][0]], [pose_3d2[0][1], pose_3d2[5][1]], zs=[pose_3d2[0][2], pose_3d2[5][2]],
                linewidth=3, color=color2)
        ax.plot([pose_3d2[0][0], pose_3d2[9][0]], [pose_3d2[0][1], pose_3d2[9][1]], zs=[pose_3d2[0][2], pose_3d2[9][2]],
                linewidth=3, color=color2)
        ax.plot([pose_3d2[0][0], pose_3d2[13][0]], [pose_3d2[0][1], pose_3d2[13][1]], zs=[pose_3d2[0][2], pose_3d2[13][2]],
                linewidth=3, color=color2)
        ax.plot([pose_3d2[0][0], pose_3d2[17][0]], [pose_3d2[0][1], pose_3d2[17][1]], zs=[pose_3d2[0][2], pose_3d2[17][2]],
                linewidth=3, color=color2)
        ax.plot([pose_3d2[1][0], pose_3d2[2][0]], [pose_3d2[1][1], pose_3d2[2][1]], zs=[pose_3d2[1][2], pose_3d2[2][2]],
                linewidth=3, color=color2)
        ax.plot([pose_3d2[2][0], pose_3d2[3][0]], [pose_3d2[2][1], pose_3d2[3][1]], zs=[pose_3d2[2][2], pose_3d2[3][2]],
                linewidth=3, color=color2)
        ax.plot([pose_3d2[3][0], pose_3d2[4][0]], [pose_3d2[3][1], pose_3d2[4][1]], zs=[pose_3d2[3][2], pose_3d2[4][2]],
                linewidth=3, color=color2)
        ax.plot([pose_3d2[5][0], pose_3d2[6][0]], [pose_3d2[5][1], pose_3d2[6][1]], zs=[pose_3d2[5][2], pose_3d2[6][2]],
                linewidth=3, color=color2)
        ax.plot([pose_3d2[6][0], pose_3d2[7][0]], [pose_3d2[6][1], pose_3d2[7][1]], zs=[pose_3d2[6][2], pose_3d2[7][2]],
                linewidth=3, color=color2)
        ax.plot([pose_3d2[7][0], pose_3d2[8][0]], [pose_3d2[7][1], pose_3d2[8][1]], zs=[pose_3d2[7][2], pose_3d2[8][2]],
                linewidth=3, color=color2)
        ax.plot([pose_3d2[9][0], pose_3d2[10][0]], [pose_3d2[9][1], pose_3d2[10][1]], zs=[pose_3d2[9][2], pose_3d2[10][2]],
                linewidth=3, color=color2)
        ax.plot([pose_3d2[10][0], pose_3d2[11][0]], [pose_3d2[10][1], pose_3d2[11][1]], zs=[pose_3d2[10][2], pose_3d2[11][2]],
                linewidth=3, color=color2)
        ax.plot([pose_3d2[11][0], pose_3d2[12][0]], [pose_3d2[11][1], pose_3d2[12][1]], zs=[pose_3d2[11][2], pose_3d2[12][2]],
                linewidth=3, color=color2)
        ax.plot([pose_3d2[13][0], pose_3d2[14][0]], [pose_3d2[13][1], pose_3d2[14][1]], zs=[pose_3d2[13][2], pose_3d2[14][2]],
                linewidth=3, color=color2)
        ax.plot([pose_3d2[14][0], pose_3d2[15][0]], [pose_3d2[14][1], pose_3d2[15][1]], zs=[pose_3d2[14][2], pose_3d2[15][2]],
                linewidth=3, color=color2)
        ax.plot([pose_3d2[15][0], pose_3d2[16][0]], [pose_3d2[15][1], pose_3d2[16][1]], zs=[pose_3d2[15][2], pose_3d2[16][2]],
                linewidth=3, color=color2)
        ax.plot([pose_3d2[17][0], pose_3d2[18][0]], [pose_3d2[17][1], pose_3d2[18][1]], zs=[pose_3d2[17][2], pose_3d2[18][2]],
                linewidth=3, color=color2)
        ax.plot([pose_3d2[18][0], pose_3d2[19][0]], [pose_3d2[18][1], pose_3d2[19][1]], zs=[pose_3d2[18][2], pose_3d2[19][2]],
                linewidth=3, color=color2)
        ax.plot([pose_3d2[19][0], pose_3d2[20][0]], [pose_3d2[19][1], pose_3d2[20][1]], zs=[pose_3d2[19][2], pose_3d2[20][2]],
                linewidth=3, color=color2)

        ax.plot([pose_3d[0][0], pose_3d[1][0]], [pose_3d[0][1], pose_3d[1][1]], zs=[pose_3d[0][2], pose_3d[1][2]],
                linewidth=3, color=b)
        ax.plot([pose_3d[0][0], pose_3d[5][0]], [pose_3d[0][1], pose_3d[5][1]], zs=[pose_3d[0][2], pose_3d[5][2]],
                linewidth=3, color=b)
        ax.plot([pose_3d[0][0], pose_3d[9][0]], [pose_3d[0][1], pose_3d[9][1]], zs=[pose_3d[0][2], pose_3d[9][2]],
                linewidth=3, color=b)
        ax.plot([pose_3d[0][0], pose_3d[13][0]], [pose_3d[0][1], pose_3d[13][1]], zs=[pose_3d[0][2], pose_3d[13][2]],
                linewidth=3, color=b)
        ax.plot([pose_3d[0][0], pose_3d[17][0]], [pose_3d[0][1], pose_3d[17][1]], zs=[pose_3d[0][2], pose_3d[17][2]],
                linewidth=3, color=b)
        ax.plot([pose_3d[1][0], pose_3d[2][0]], [pose_3d[1][1], pose_3d[2][1]], zs=[pose_3d[1][2], pose_3d[2][2]],
                linewidth=3, color=b)
        ax.plot([pose_3d[2][0], pose_3d[3][0]], [pose_3d[2][1], pose_3d[3][1]], zs=[pose_3d[2][2], pose_3d[3][2]],
                linewidth=3, color=b)
        ax.plot([pose_3d[3][0], pose_3d[4][0]], [pose_3d[3][1], pose_3d[4][1]], zs=[pose_3d[3][2], pose_3d[4][2]],
                linewidth=3, color=b)
        ax.plot([pose_3d[5][0], pose_3d[6][0]], [pose_3d[5][1], pose_3d[6][1]], zs=[pose_3d[5][2], pose_3d[6][2]],
                linewidth=3, color=b)
        ax.plot([pose_3d[6][0], pose_3d[7][0]], [pose_3d[6][1], pose_3d[7][1]], zs=[pose_3d[6][2], pose_3d[7][2]],
                linewidth=3, color=b)
        ax.plot([pose_3d[7][0], pose_3d[8][0]], [pose_3d[7][1], pose_3d[8][1]], zs=[pose_3d[7][2], pose_3d[8][2]],
                linewidth=3, color=b)
        ax.plot([pose_3d[9][0], pose_3d[10][0]], [pose_3d[9][1], pose_3d[10][1]], zs=[pose_3d[9][2], pose_3d[10][2]],
                linewidth=3, color=b)
        ax.plot([pose_3d[10][0], pose_3d[11][0]], [pose_3d[10][1], pose_3d[11][1]], zs=[pose_3d[10][2], pose_3d[11][2]],
                linewidth=3, color=b)
        ax.plot([pose_3d[11][0], pose_3d[12][0]], [pose_3d[11][1], pose_3d[12][1]], zs=[pose_3d[11][2], pose_3d[12][2]],
                linewidth=3, color=b)
        ax.plot([pose_3d[13][0], pose_3d[14][0]], [pose_3d[13][1], pose_3d[14][1]], zs=[pose_3d[13][2], pose_3d[14][2]],
                linewidth=3, color=b)
        ax.plot([pose_3d[14][0], pose_3d[15][0]], [pose_3d[14][1], pose_3d[15][1]], zs=[pose_3d[14][2], pose_3d[15][2]],
                linewidth=3, color=b)
        ax.plot([pose_3d[15][0], pose_3d[16][0]], [pose_3d[15][1], pose_3d[16][1]], zs=[pose_3d[15][2], pose_3d[16][2]],
                linewidth=3, color=b)
        ax.plot([pose_3d[17][0], pose_3d[18][0]], [pose_3d[17][1], pose_3d[18][1]], zs=[pose_3d[17][2], pose_3d[18][2]],
                linewidth=3, color=b)
        ax.plot([pose_3d[18][0], pose_3d[19][0]], [pose_3d[18][1], pose_3d[19][1]], zs=[pose_3d[18][2], pose_3d[19][2]],
                linewidth=3, color=b)
        ax.plot([pose_3d[19][0], pose_3d[20][0]], [pose_3d[19][1], pose_3d[20][1]], zs=[pose_3d[19][2], pose_3d[20][2]],
                linewidth=3, color=b)
        plt.show()
        return ax




def getHandMask(y_hat, mask):
    batch_size, seq_size, _, h, w = mask.size()
    loss_mask = torch.ones(batch_size, seq_size, 1).cuda()
    y_hat = y_hat.round().long()
    y_hat[:, :, :, 0] = torch.where(y_hat[:, :, :, 0] >= w, torch.tensor(w-1, dtype=torch.long).cuda(), y_hat[:, :, :, 0])
    y_hat[:, :, :, 1] = torch.where(y_hat[:, :, :, 1] >= h, torch.tensor(h-1, dtype=torch.long).cuda(), y_hat[:, :, :, 1])
    y_hat[:, :, :, 0] = torch.where(y_hat[:, :, :, 0] < 0, torch.tensor(0, dtype=torch.long).cuda(), y_hat[:, :, :, 0])
    y_hat[:, :, :, 1] = torch.where(y_hat[:, :, :, 1] < 0, torch.tensor(0, dtype=torch.long).cuda(), y_hat[:, :, :, 1])
    for i_batch in range(batch_size):
        for i_seq in range(seq_size):
            loss_mask[i_batch, i_seq] = loss_mask[i_batch, i_seq] - mask[i_batch, i_seq, 0, y_hat[i_batch, i_seq, :, 1], y_hat[i_batch, i_seq, :, 0]].sum()/778
    return loss_mask.mean()



def getHandMask(y_hat, mask):
    batch_size, seq_size, _, h, w = mask.size()
    loss_mask = torch.ones(batch_size, seq_size, 1).cuda()
    y_hat = y_hat.round().long()
    y_hat[:, :, :, 0] = torch.where(y_hat[:, :, :, 0] >= w, torch.tensor(w-1, dtype=torch.long).cuda(), y_hat[:, :, :, 0])
    y_hat[:, :, :, 1] = torch.where(y_hat[:, :, :, 1] >= h, torch.tensor(h-1, dtype=torch.long).cuda(), y_hat[:, :, :, 1])
    y_hat[:, :, :, 0] = torch.where(y_hat[:, :, :, 0] < 0, torch.tensor(0, dtype=torch.long).cuda(), y_hat[:, :, :, 0])
    y_hat[:, :, :, 1] = torch.where(y_hat[:, :, :, 1] < 0, torch.tensor(0, dtype=torch.long).cuda(), y_hat[:, :, :, 1])
    for i_batch in range(batch_size):
        for i_seq in range(seq_size):
            loss_mask[i_batch, i_seq] = loss_mask[i_batch, i_seq] - mask[i_batch, i_seq, 0, y_hat[i_batch, i_seq, :, 1], y_hat[i_batch, i_seq, :, 0]].sum()/778
    return loss_mask.mean()


import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
class STBLoss(nn.Module):
    def __init__(self, args, pretrain=False):
        super(STBLoss, self).__init__()
        # Initialize Parameters
        self.args = args
        self.pretrain = pretrain
        self.alpha_2d = 1
        self.alpha_3d = 10 #100
        self.alpha_mask = 5 # 100
        self.alpha_reg = 0#10
        self.alpha_beta = 0#10000
        self.alpha_camera = 1
        self.n_meshes = 778
        self.img_size = 224

    def getRampUpScale(self, epoch):
        if self.pretrain:
            return torch.ones(1).cuda()
        else:
            return torch.ones(1).cuda()
            # return torch.FloatTensor([(epoch+1) / self.args.max_epochs_ramp_up]).cuda()

    def forward(self, epoch, mask, predictions, targets):
        # Initialize predictions
        x2d_pred, x3d_pred, camera_param_pred, theta, beta = predictions
        # Initialize targets
        joint_2d_target, joint_3d_target, verts_3d_target, camera_param_target, dataset_type = targets
        # Initialize Variables
        batch_size, seq_size, _ = x2d_pred.size()
        # Get Vectors
        joint_2d_pred = torch.stack((x2d_pred[:, :, :42:2], x2d_pred[:, :, 1:42:2]), dim=3)  # x_hat
        y_hat = x2d_pred[:, :, 42:].view(batch_size, seq_size, 778, 2)
        joint_3d_pred = torch.stack((x3d_pred[:, :, :63:3], x3d_pred[:, :, 1:63:3], x3d_pred[:, :, 2:63:3]), dim=3)
        verts_3d_pred = torch.stack((x3d_pred[:, :, 63::3], x3d_pred[:, :, 64::3], x3d_pred[:, :, 65::3]), dim=3)
        # Calculate the Losses - 2D joint re-projection loss
        loss_2d = torch.abs((joint_2d_pred.view(batch_size*seq_size, -1) / self.img_size - joint_2d_target.view(batch_size*seq_size, -1) / self.img_size)).sum(1).mean()
        loss_2d = self.alpha_2d * loss_2d
        # Calculate the Losses - Temporal loss
        loss_temp = torch.zeros(1).cuda()
        # theta_prev = torch.cat((theta[:, 0, :].unsqueeze(1), theta[:, :-1, :]), 1)
        # beta_prev = torch.cat((beta[:, 0, :].unsqueeze(1), beta[:, :-1:]), 1)
        # pose_temp_loss = torch.pow(theta_prev.view(batch_size * seq_size, -1) - theta.view(batch_size * seq_size, -1),
        #                            2).sum(1).mean()
        # shape_temp_loss = torch.pow(beta_prev.view(batch_size * seq_size, -1) - beta.view(batch_size * seq_size, -1),
        #                             2).sum(1).mean()
        # loss_temp = 0.1 * pose_temp_loss + 1. * shape_temp_loss
        # Calculate the Losses - Hand mask loss
        loss_mask = self.alpha_mask * getHandMask(y_hat, mask)
        # Calculate the Losses - Camera loss
        loss_camera = torch.zeros(1).cuda()
        # Calculate the Losses - Regularization loss
        loss_reg = torch.zeros(1).cuda()
        # Calculate the Losses - 3D joint loss (Only the STEREO dataset)
        # diff_3d = joint_3d_pred.view(batch_size * seq_size, -1, 3) - joint_3d_target.view(batch_size * seq_size, -1, 3)
        ## normalize

        joint_3d_pred, pred_min, pred_max = self.normalize_joints_scale(joint_3d_pred)
        joint_3d_target, targ_min, targ_max = self.normalize_joints_scale(joint_3d_target)
        verts_3d_pred, _, pred_max_v = self.normalize_joints_scale(verts_3d_pred)
        _, _, maxp = self.normalize_joints_scale(joint_3d_pred)
        _, _, maxt = self.normalize_joints_scale(joint_3d_target)
        _, _, maxv = self.normalize_joints_scale(verts_3d_pred)
        # _, pred_mean, pred_std = self.normalize_joints(norm_scaled_joint3d_pred)
        # # verts_3d_pred, _, _ = self.normalize_joints(verts_3d_pred)
        # joint_3d_target, _, _ = self.normalize_joints(joint_3d_target)
        # print(pred_max.size(), pred_mean.size())
        # joint_3d_target = self.denormalize_joints_scale(joint_3d_target, pred_min, pred_max)
        joint_3d_pred = self.center_joints_scale(joint_3d_pred, maxp)
        joint_3d_target = self.center_joints_scale(joint_3d_target,maxt )
        verts_3d_pred =  self.center_joints_scale(verts_3d_pred, maxv)
        # ax = self.draw_3d_mano_pose(joint_3d_pred[0][0], joint_3d_target[0][0], 'black', 'red')
        # ax.scatter(xs=verts_3d_pred[0,0,:, 0].cpu().numpy(), ys=verts_3d_pred[0,0,:, 1].cpu().numpy(), zs=verts_3d_pred[0, 0, :, 2].cpu().numpy())
        # plt.show()
        palm_cent_pred = 0.5 * verts_3d_pred[:, :, 17, :] + 0.5 * verts_3d_pred[:, :, 67, :]
        palm_cent_targ = joint_3d_target[:, :, 0, :]
        diff_3d_pc = palm_cent_pred.view(batch_size * seq_size, 3) - palm_cent_targ.view(batch_size * seq_size, 3)
        diff_3d_ot = joint_3d_pred[:, :, 1:, :].view(batch_size * seq_size, -1, 3) - joint_3d_target[:, :, 1:, :].view(batch_size * seq_size, -1, 3)
        diff_3d = torch.cat((diff_3d_pc.unsqueeze(1), diff_3d_ot), 1)
        loss_3d = self.alpha_3d * torch.pow(diff_3d.view(batch_size * seq_size, -1), 2).sum(1).mean()
        diff_3d = diff_3d * (pred_max - pred_min).repeat(1,1,21,1).view(batch_size * seq_size, -1, 3)
        # Weighted sum
        loss = loss_2d + loss_3d + loss_mask + loss_reg + loss_camera + loss_temp

        # Initialize Average Distance Storage
        avg_distance_2d = list()
        avg_distance_3d = list()
        for _ in range(self.args.n_kps):
            avg_distance_2d.append(None)
            avg_distance_3d.append(None)
        # Calculate euclidean distance
        diff_2d = joint_2d_pred.view(batch_size*seq_size, -1, 2) - joint_2d_target.view(batch_size*seq_size, -1, 2)
        euclidean_dist_2d = np.sqrt(np.sum(np.square(diff_2d.detach().cpu().numpy()), axis=2))
        euclidean_dist_3d = np.sqrt(np.sum(np.square(diff_3d.detach().cpu().numpy()), axis=2))
        for i in range(self.args.n_kps):
            avg_distance_2d[i] = euclidean_dist_2d[:, i]
            avg_distance_3d[i] = euclidean_dist_3d[:, i]
        return loss, [loss_2d.item(), loss_3d.item(), loss_mask.item(), loss_reg.item(), loss_camera.item(), avg_distance_2d, avg_distance_3d]



    def normalize_joints_scale(self, hand_joints):
        min_joints, _ = torch.min(hand_joints, dim=2, keepdim=True)
        max_joints, _ = torch.max(hand_joints, dim=2, keepdim=True)
        hand_joints[:, :, :, 0] = (hand_joints[:, :, :, 0] - min_joints[:, :, :, 0]) / (max_joints[:, :, :, 0] - min_joints[:, :, :, 0])
        hand_joints[:, :, :, 1] = (hand_joints[:, :, :, 1] - min_joints[:, :, :, 1]) / (max_joints[:, :, :, 0] - min_joints[:, :, :, 0])
        hand_joints[:, :, :, 2] = (hand_joints[:, :, :, 2] - min_joints[:, :, :, 2]) / (max_joints[:, :, :, 0] - min_joints[:, :, :, 0])
        return hand_joints, min_joints, max_joints


    def center_joints_scale(self, hand_joints, max_joints):
        hand_joints[:, :, :, 0] = hand_joints[:, :, :, 0]  - max_joints[:, :, :, 0]
        hand_joints[:, :, :, 1] =  hand_joints[:, :, :, 1]   -max_joints[:, :, :, 1]
        hand_joints[:, :, :, 2] =  hand_joints[:, :, :, 2]  - max_joints[:, :, :, 2]
        return hand_joints


    def draw_3d_mano_pose(self, pose_3d, pose_3d2, color='black', color2 ='red'):
        pose_3d = pose_3d.reshape(21, 3)
        pose_3d2 = pose_3d2.reshape(21, 3)
        # print(pose_3d[0][0], pose_3d[0][1], pose_3d[0][2])
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        b = color # or 'red'
        ax.plot([pose_3d2[0][0], pose_3d2[1][0]], [pose_3d2[0][1], pose_3d2[1][1]], zs=[pose_3d2[0][2], pose_3d2[1][2]],
                linewidth=3, color=color2)
        ax.plot([pose_3d2[0][0], pose_3d2[5][0]], [pose_3d2[0][1], pose_3d2[5][1]], zs=[pose_3d2[0][2], pose_3d2[5][2]],
                linewidth=3, color=color2)
        ax.plot([pose_3d2[0][0], pose_3d2[9][0]], [pose_3d2[0][1], pose_3d2[9][1]], zs=[pose_3d2[0][2], pose_3d2[9][2]],
                linewidth=3, color=color2)
        ax.plot([pose_3d2[0][0], pose_3d2[13][0]], [pose_3d2[0][1], pose_3d2[13][1]], zs=[pose_3d2[0][2], pose_3d2[13][2]],
                linewidth=3, color=color2)
        ax.plot([pose_3d2[0][0], pose_3d2[17][0]], [pose_3d2[0][1], pose_3d2[17][1]], zs=[pose_3d2[0][2], pose_3d2[17][2]],
                linewidth=3, color=color2)
        ax.plot([pose_3d2[1][0], pose_3d2[2][0]], [pose_3d2[1][1], pose_3d2[2][1]], zs=[pose_3d2[1][2], pose_3d2[2][2]],
                linewidth=3, color=color2)
        ax.plot([pose_3d2[2][0], pose_3d2[3][0]], [pose_3d2[2][1], pose_3d2[3][1]], zs=[pose_3d2[2][2], pose_3d2[3][2]],
                linewidth=3, color=color2)
        ax.plot([pose_3d2[3][0], pose_3d2[4][0]], [pose_3d2[3][1], pose_3d2[4][1]], zs=[pose_3d2[3][2], pose_3d2[4][2]],
                linewidth=3, color=color2)
        ax.plot([pose_3d2[5][0], pose_3d2[6][0]], [pose_3d2[5][1], pose_3d2[6][1]], zs=[pose_3d2[5][2], pose_3d2[6][2]],
                linewidth=3, color=color2)
        ax.plot([pose_3d2[6][0], pose_3d2[7][0]], [pose_3d2[6][1], pose_3d2[7][1]], zs=[pose_3d2[6][2], pose_3d2[7][2]],
                linewidth=3, color=color2)
        ax.plot([pose_3d2[7][0], pose_3d2[8][0]], [pose_3d2[7][1], pose_3d2[8][1]], zs=[pose_3d2[7][2], pose_3d2[8][2]],
                linewidth=3, color=color2)
        ax.plot([pose_3d2[9][0], pose_3d2[10][0]], [pose_3d2[9][1], pose_3d2[10][1]], zs=[pose_3d2[9][2], pose_3d2[10][2]],
                linewidth=3, color=color2)
        ax.plot([pose_3d2[10][0], pose_3d2[11][0]], [pose_3d2[10][1], pose_3d2[11][1]], zs=[pose_3d2[10][2], pose_3d2[11][2]],
                linewidth=3, color=color2)
        ax.plot([pose_3d2[11][0], pose_3d2[12][0]], [pose_3d2[11][1], pose_3d2[12][1]], zs=[pose_3d2[11][2], pose_3d2[12][2]],
                linewidth=3, color=color2)
        ax.plot([pose_3d2[13][0], pose_3d2[14][0]], [pose_3d2[13][1], pose_3d2[14][1]], zs=[pose_3d2[13][2], pose_3d2[14][2]],
                linewidth=3, color=color2)
        ax.plot([pose_3d2[14][0], pose_3d2[15][0]], [pose_3d2[14][1], pose_3d2[15][1]], zs=[pose_3d2[14][2], pose_3d2[15][2]],
                linewidth=3, color=color2)
        ax.plot([pose_3d2[15][0], pose_3d2[16][0]], [pose_3d2[15][1], pose_3d2[16][1]], zs=[pose_3d2[15][2], pose_3d2[16][2]],
                linewidth=3, color=color2)
        ax.plot([pose_3d2[17][0], pose_3d2[18][0]], [pose_3d2[17][1], pose_3d2[18][1]], zs=[pose_3d2[17][2], pose_3d2[18][2]],
                linewidth=3, color=color2)
        ax.plot([pose_3d2[18][0], pose_3d2[19][0]], [pose_3d2[18][1], pose_3d2[19][1]], zs=[pose_3d2[18][2], pose_3d2[19][2]],
                linewidth=3, color=color2)
        ax.plot([pose_3d2[19][0], pose_3d2[20][0]], [pose_3d2[19][1], pose_3d2[20][1]], zs=[pose_3d2[19][2], pose_3d2[20][2]],
                linewidth=3, color=color2)

        ax.plot([pose_3d[0][0], pose_3d[1][0]], [pose_3d[0][1], pose_3d[1][1]], zs=[pose_3d[0][2], pose_3d[1][2]],
                linewidth=3, color=b)
        ax.plot([pose_3d[0][0], pose_3d[5][0]], [pose_3d[0][1], pose_3d[5][1]], zs=[pose_3d[0][2], pose_3d[5][2]],
                linewidth=3, color=b)
        ax.plot([pose_3d[0][0], pose_3d[9][0]], [pose_3d[0][1], pose_3d[9][1]], zs=[pose_3d[0][2], pose_3d[9][2]],
                linewidth=3, color=b)
        ax.plot([pose_3d[0][0], pose_3d[13][0]], [pose_3d[0][1], pose_3d[13][1]], zs=[pose_3d[0][2], pose_3d[13][2]],
                linewidth=3, color=b)
        ax.plot([pose_3d[0][0], pose_3d[17][0]], [pose_3d[0][1], pose_3d[17][1]], zs=[pose_3d[0][2], pose_3d[17][2]],
                linewidth=3, color=b)
        ax.plot([pose_3d[1][0], pose_3d[2][0]], [pose_3d[1][1], pose_3d[2][1]], zs=[pose_3d[1][2], pose_3d[2][2]],
                linewidth=3, color=b)
        ax.plot([pose_3d[2][0], pose_3d[3][0]], [pose_3d[2][1], pose_3d[3][1]], zs=[pose_3d[2][2], pose_3d[3][2]],
                linewidth=3, color=b)
        ax.plot([pose_3d[3][0], pose_3d[4][0]], [pose_3d[3][1], pose_3d[4][1]], zs=[pose_3d[3][2], pose_3d[4][2]],
                linewidth=3, color=b)
        ax.plot([pose_3d[5][0], pose_3d[6][0]], [pose_3d[5][1], pose_3d[6][1]], zs=[pose_3d[5][2], pose_3d[6][2]],
                linewidth=3, color=b)
        ax.plot([pose_3d[6][0], pose_3d[7][0]], [pose_3d[6][1], pose_3d[7][1]], zs=[pose_3d[6][2], pose_3d[7][2]],
                linewidth=3, color=b)
        ax.plot([pose_3d[7][0], pose_3d[8][0]], [pose_3d[7][1], pose_3d[8][1]], zs=[pose_3d[7][2], pose_3d[8][2]],
                linewidth=3, color=b)
        ax.plot([pose_3d[9][0], pose_3d[10][0]], [pose_3d[9][1], pose_3d[10][1]], zs=[pose_3d[9][2], pose_3d[10][2]],
                linewidth=3, color=b)
        ax.plot([pose_3d[10][0], pose_3d[11][0]], [pose_3d[10][1], pose_3d[11][1]], zs=[pose_3d[10][2], pose_3d[11][2]],
                linewidth=3, color=b)
        ax.plot([pose_3d[11][0], pose_3d[12][0]], [pose_3d[11][1], pose_3d[12][1]], zs=[pose_3d[11][2], pose_3d[12][2]],
                linewidth=3, color=b)
        ax.plot([pose_3d[13][0], pose_3d[14][0]], [pose_3d[13][1], pose_3d[14][1]], zs=[pose_3d[13][2], pose_3d[14][2]],
                linewidth=3, color=b)
        ax.plot([pose_3d[14][0], pose_3d[15][0]], [pose_3d[14][1], pose_3d[15][1]], zs=[pose_3d[14][2], pose_3d[15][2]],
                linewidth=3, color=b)
        ax.plot([pose_3d[15][0], pose_3d[16][0]], [pose_3d[15][1], pose_3d[16][1]], zs=[pose_3d[15][2], pose_3d[16][2]],
                linewidth=3, color=b)
        ax.plot([pose_3d[17][0], pose_3d[18][0]], [pose_3d[17][1], pose_3d[18][1]], zs=[pose_3d[17][2], pose_3d[18][2]],
                linewidth=3, color=b)
        ax.plot([pose_3d[18][0], pose_3d[19][0]], [pose_3d[18][1], pose_3d[19][1]], zs=[pose_3d[18][2], pose_3d[19][2]],
                linewidth=3, color=b)
        ax.plot([pose_3d[19][0], pose_3d[20][0]], [pose_3d[19][1], pose_3d[20][1]], zs=[pose_3d[19][2], pose_3d[20][2]],
                linewidth=3, color=b)
        return ax





class RWorldLoss(nn.Module):
    def __init__(self, args, pretrain=False):
        super(RWorldLoss, self).__init__()
        # Initialize Parameters
        self.args = args
        self.pretrain = pretrain
        if self.pretrain:
            self.alpha_2d = 0
            self.alpha_3d = 0
            self.alpha_mask = 0
            self.alpha_reg = 0
            self.alpha_beta = 0
            self.alpha_camera = 1
        else:
            self.alpha_2d = 5
            self.alpha_3d = 100
            self.alpha_mask = 0
            self.alpha_reg = 0
            self.alpha_beta = 0
            self.alpha_camera = 0
        self.n_meshes = 778
        self.img_size = 224

    def getRampUpScale(self, epoch):
        if self.pretrain:
            return torch.ones(1).cuda()
        else:
            return torch.ones(1).cuda()
            # return torch.FloatTensor([(epoch+1) / self.args.max_epochs_ramp_up]).cuda()

    def forward(self, epoch, mask, predictions, targets):
        # Initialize predictions
        x2d_pred, x3d_pred, camera_param_pred, theta, beta = predictions
        # Initialize targets
        joint_2d_target, joint_3d_target, verts_3d_target, camera_param_target, dataset_type = targets
        # print(joint_3d_target, x3d_pred)
        batch_size, seq_size, _ = x2d_pred.size()
        # Get Vectors
        joint_2d_pred = torch.stack((x2d_pred[:, :, :42:2], x2d_pred[:, :, 1:42:2]), dim=3)  # x_hat
        y_hat = x2d_pred[:, :, 42:].view(batch_size, seq_size, 778, 2)
        # No loss for Camera param vert3d loss,
        joint_3d_pred = torch.stack((x3d_pred[:, :, :63:3], x3d_pred[:, :, 1:63:3], x3d_pred[:, :, 2:63:3]), dim=3)  # out2[:, :21, :]

        joint_3d_pred, pred_min, pred_max = self.normalize_joints_scale(joint_3d_pred)
        joint_3d_target, targ_min, targ_max = self.normalize_joints_scale(joint_3d_target)
        _, _, maxp = self.normalize_joints_scale(joint_3d_pred)
        _, _, maxt = self.normalize_joints_scale(joint_3d_target)
        joint_3d_pred = self.center_joints_scale(joint_3d_pred, maxp)
        joint_3d_target = self.center_joints_scale(joint_3d_target, maxt)
        diff_3d = joint_3d_pred.view(batch_size * seq_size, -1, 3) - joint_3d_target.view(batch_size * seq_size, -1, 3)
        loss_3d = self.alpha_3d * torch.pow(diff_3d.view(batch_size * seq_size, -1), 2).sum(1).mean()
        diff_3d = diff_3d * (pred_max - pred_min).repeat(1, 1, 21, 1).view(batch_size * seq_size, -1, 3)
        # Weighted sum
        loss_2d = torch.abs((joint_2d_pred.view(batch_size * seq_size, -1) / self.img_size - joint_2d_target.view(
            batch_size * seq_size, -1) / self.img_size)).sum(1).mean()
        loss_2d = self.alpha_2d * loss_2d
        loss_temp = torch.zeros(1).cuda()
        loss_mask = self.alpha_mask * getHandMask(y_hat, mask)
        loss_camera = torch.zeros(1).cuda()
        loss_reg = torch.zeros(1).cuda()
        loss = loss_2d + loss_3d + loss_mask + loss_reg + loss_camera + loss_temp

        # Initialize Average Distance Storage
        avg_distance_2d = list()
        avg_distance_3d = list()
        for _ in range(self.args.n_kps):
            avg_distance_2d.append(None)
            avg_distance_3d.append(None)
        # Calculate euclidean distance
        diff_2d = joint_2d_pred.view(batch_size * seq_size, -1, 2) - joint_2d_target.view(batch_size * seq_size, -1, 2)
        euclidean_dist_2d = np.sqrt(np.sum(np.square(diff_2d.detach().cpu().numpy()), axis=2))
        euclidean_dist_3d = np.sqrt(np.sum(np.square(diff_3d.detach().cpu().numpy()), axis=2))
        for i in range(self.args.n_kps):
            avg_distance_2d[i] = euclidean_dist_2d[:, i]
            avg_distance_3d[i] = euclidean_dist_3d[:, i]
        return loss, [loss_2d.item(), loss_3d.item(), loss_mask.item(), loss_reg.item(), loss_camera.item(),
                      avg_distance_2d, avg_distance_3d]

    def getHandMask(self, y_hat, mask):
        batch_size, seq_size, _, h, w = mask.size()
        loss_mask = torch.ones(batch_size, seq_size, 1).cuda()
        y_hat = y_hat.round().long()
        y_hat[:, :, :, 0] = torch.where(y_hat[:, :, :, 0] >= w, torch.tensor(w-1, dtype=torch.long).cuda(), y_hat[:, :, :, 0])
        y_hat[:, :, :, 1] = torch.where(y_hat[:, :, :, 1] >= h, torch.tensor(h-1, dtype=torch.long).cuda(), y_hat[:, :, :, 1])
        y_hat[:, :, :, 0] = torch.where(y_hat[:, :, :, 0] < 0, torch.tensor(0, dtype=torch.long).cuda(), y_hat[:, :, :, 0])
        y_hat[:, :, :, 1] = torch.where(y_hat[:, :, :, 1] < 0, torch.tensor(0, dtype=torch.long).cuda(), y_hat[:, :, :, 1])
        for i_batch in range(batch_size):
            for i_seq in range(seq_size):
                loss_mask[i_batch, i_seq] = loss_mask[i_batch, i_seq] - mask[i_batch, i_seq, 0, y_hat[i_batch, i_seq, :, 1], y_hat[i_batch, i_seq, :, 0]].sum()/self.n_meshes
        return loss_mask.mean()

    def normalize_joints_scale(self, hand_joints):
        min_joints, _ = torch.min(hand_joints, dim=2, keepdim=True)
        max_joints, _ = torch.max(hand_joints, dim=2, keepdim=True)
        hand_joints[:, :, :, 0] = (hand_joints[:, :, :, 0] - min_joints[:, :, :, 0]) / (max_joints[:, :, :, 0] - min_joints[:, :, :, 0])
        hand_joints[:, :, :, 1] = (hand_joints[:, :, :, 1] - min_joints[:, :, :, 1]) / (max_joints[:, :, :, 0] - min_joints[:, :, :, 0])
        hand_joints[:, :, :, 2] = (hand_joints[:, :, :, 2] - min_joints[:, :, :, 2]) / (max_joints[:, :, :, 0] - min_joints[:, :, :, 0])
        return hand_joints, min_joints, max_joints


    def center_joints_scale(self, hand_joints, max_joints):
        hand_joints[:, :, :, 0] = hand_joints[:, :, :, 0]  - max_joints[:, :, :, 0]
        hand_joints[:, :, :, 1] =  hand_joints[:, :, :, 1]   -max_joints[:, :, :, 1]
        hand_joints[:, :, :, 2] =  hand_joints[:, :, :, 2]  - max_joints[:, :, :, 2]
        return hand_joints


    def draw_3d_mano_pose(self, pose_3d, pose_3d2, color='black', color2 ='red'):
        pose_3d = pose_3d.reshape(21, 3)
        pose_3d2 = pose_3d2.reshape(21, 3)
        # print(pose_3d[0][0], pose_3d[0][1], pose_3d[0][2])
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        b = color # or 'red'
        ax.plot([pose_3d2[0][0], pose_3d2[1][0]], [pose_3d2[0][1], pose_3d2[1][1]], zs=[pose_3d2[0][2], pose_3d2[1][2]],
                linewidth=3, color=color2)
        ax.plot([pose_3d2[0][0], pose_3d2[5][0]], [pose_3d2[0][1], pose_3d2[5][1]], zs=[pose_3d2[0][2], pose_3d2[5][2]],
                linewidth=3, color=color2)
        ax.plot([pose_3d2[0][0], pose_3d2[9][0]], [pose_3d2[0][1], pose_3d2[9][1]], zs=[pose_3d2[0][2], pose_3d2[9][2]],
                linewidth=3, color=color2)
        ax.plot([pose_3d2[0][0], pose_3d2[13][0]], [pose_3d2[0][1], pose_3d2[13][1]], zs=[pose_3d2[0][2], pose_3d2[13][2]],
                linewidth=3, color=color2)
        ax.plot([pose_3d2[0][0], pose_3d2[17][0]], [pose_3d2[0][1], pose_3d2[17][1]], zs=[pose_3d2[0][2], pose_3d2[17][2]],
                linewidth=3, color=color2)
        ax.plot([pose_3d2[1][0], pose_3d2[2][0]], [pose_3d2[1][1], pose_3d2[2][1]], zs=[pose_3d2[1][2], pose_3d2[2][2]],
                linewidth=3, color=color2)
        ax.plot([pose_3d2[2][0], pose_3d2[3][0]], [pose_3d2[2][1], pose_3d2[3][1]], zs=[pose_3d2[2][2], pose_3d2[3][2]],
                linewidth=3, color=color2)
        ax.plot([pose_3d2[3][0], pose_3d2[4][0]], [pose_3d2[3][1], pose_3d2[4][1]], zs=[pose_3d2[3][2], pose_3d2[4][2]],
                linewidth=3, color=color2)
        ax.plot([pose_3d2[5][0], pose_3d2[6][0]], [pose_3d2[5][1], pose_3d2[6][1]], zs=[pose_3d2[5][2], pose_3d2[6][2]],
                linewidth=3, color=color2)
        ax.plot([pose_3d2[6][0], pose_3d2[7][0]], [pose_3d2[6][1], pose_3d2[7][1]], zs=[pose_3d2[6][2], pose_3d2[7][2]],
                linewidth=3, color=color2)
        ax.plot([pose_3d2[7][0], pose_3d2[8][0]], [pose_3d2[7][1], pose_3d2[8][1]], zs=[pose_3d2[7][2], pose_3d2[8][2]],
                linewidth=3, color=color2)
        ax.plot([pose_3d2[9][0], pose_3d2[10][0]], [pose_3d2[9][1], pose_3d2[10][1]], zs=[pose_3d2[9][2], pose_3d2[10][2]],
                linewidth=3, color=color2)
        ax.plot([pose_3d2[10][0], pose_3d2[11][0]], [pose_3d2[10][1], pose_3d2[11][1]], zs=[pose_3d2[10][2], pose_3d2[11][2]],
                linewidth=3, color=color2)
        ax.plot([pose_3d2[11][0], pose_3d2[12][0]], [pose_3d2[11][1], pose_3d2[12][1]], zs=[pose_3d2[11][2], pose_3d2[12][2]],
                linewidth=3, color=color2)
        ax.plot([pose_3d2[13][0], pose_3d2[14][0]], [pose_3d2[13][1], pose_3d2[14][1]], zs=[pose_3d2[13][2], pose_3d2[14][2]],
                linewidth=3, color=color2)
        ax.plot([pose_3d2[14][0], pose_3d2[15][0]], [pose_3d2[14][1], pose_3d2[15][1]], zs=[pose_3d2[14][2], pose_3d2[15][2]],
                linewidth=3, color=color2)
        ax.plot([pose_3d2[15][0], pose_3d2[16][0]], [pose_3d2[15][1], pose_3d2[16][1]], zs=[pose_3d2[15][2], pose_3d2[16][2]],
                linewidth=3, color=color2)
        ax.plot([pose_3d2[17][0], pose_3d2[18][0]], [pose_3d2[17][1], pose_3d2[18][1]], zs=[pose_3d2[17][2], pose_3d2[18][2]],
                linewidth=3, color=color2)
        ax.plot([pose_3d2[18][0], pose_3d2[19][0]], [pose_3d2[18][1], pose_3d2[19][1]], zs=[pose_3d2[18][2], pose_3d2[19][2]],
                linewidth=3, color=color2)
        ax.plot([pose_3d2[19][0], pose_3d2[20][0]], [pose_3d2[19][1], pose_3d2[20][1]], zs=[pose_3d2[19][2], pose_3d2[20][2]],
                linewidth=3, color=color2)

        ax.plot([pose_3d[0][0], pose_3d[1][0]], [pose_3d[0][1], pose_3d[1][1]], zs=[pose_3d[0][2], pose_3d[1][2]],
                linewidth=3, color=b)
        ax.plot([pose_3d[0][0], pose_3d[5][0]], [pose_3d[0][1], pose_3d[5][1]], zs=[pose_3d[0][2], pose_3d[5][2]],
                linewidth=3, color=b)
        ax.plot([pose_3d[0][0], pose_3d[9][0]], [pose_3d[0][1], pose_3d[9][1]], zs=[pose_3d[0][2], pose_3d[9][2]],
                linewidth=3, color=b)
        ax.plot([pose_3d[0][0], pose_3d[13][0]], [pose_3d[0][1], pose_3d[13][1]], zs=[pose_3d[0][2], pose_3d[13][2]],
                linewidth=3, color=b)
        ax.plot([pose_3d[0][0], pose_3d[17][0]], [pose_3d[0][1], pose_3d[17][1]], zs=[pose_3d[0][2], pose_3d[17][2]],
                linewidth=3, color=b)
        ax.plot([pose_3d[1][0], pose_3d[2][0]], [pose_3d[1][1], pose_3d[2][1]], zs=[pose_3d[1][2], pose_3d[2][2]],
                linewidth=3, color=b)
        ax.plot([pose_3d[2][0], pose_3d[3][0]], [pose_3d[2][1], pose_3d[3][1]], zs=[pose_3d[2][2], pose_3d[3][2]],
                linewidth=3, color=b)
        ax.plot([pose_3d[3][0], pose_3d[4][0]], [pose_3d[3][1], pose_3d[4][1]], zs=[pose_3d[3][2], pose_3d[4][2]],
                linewidth=3, color=b)
        ax.plot([pose_3d[5][0], pose_3d[6][0]], [pose_3d[5][1], pose_3d[6][1]], zs=[pose_3d[5][2], pose_3d[6][2]],
                linewidth=3, color=b)
        ax.plot([pose_3d[6][0], pose_3d[7][0]], [pose_3d[6][1], pose_3d[7][1]], zs=[pose_3d[6][2], pose_3d[7][2]],
                linewidth=3, color=b)
        ax.plot([pose_3d[7][0], pose_3d[8][0]], [pose_3d[7][1], pose_3d[8][1]], zs=[pose_3d[7][2], pose_3d[8][2]],
                linewidth=3, color=b)
        ax.plot([pose_3d[9][0], pose_3d[10][0]], [pose_3d[9][1], pose_3d[10][1]], zs=[pose_3d[9][2], pose_3d[10][2]],
                linewidth=3, color=b)
        ax.plot([pose_3d[10][0], pose_3d[11][0]], [pose_3d[10][1], pose_3d[11][1]], zs=[pose_3d[10][2], pose_3d[11][2]],
                linewidth=3, color=b)
        ax.plot([pose_3d[11][0], pose_3d[12][0]], [pose_3d[11][1], pose_3d[12][1]], zs=[pose_3d[11][2], pose_3d[12][2]],
                linewidth=3, color=b)
        ax.plot([pose_3d[13][0], pose_3d[14][0]], [pose_3d[13][1], pose_3d[14][1]], zs=[pose_3d[13][2], pose_3d[14][2]],
                linewidth=3, color=b)
        ax.plot([pose_3d[14][0], pose_3d[15][0]], [pose_3d[14][1], pose_3d[15][1]], zs=[pose_3d[14][2], pose_3d[15][2]],
                linewidth=3, color=b)
        ax.plot([pose_3d[15][0], pose_3d[16][0]], [pose_3d[15][1], pose_3d[16][1]], zs=[pose_3d[15][2], pose_3d[16][2]],
                linewidth=3, color=b)
        ax.plot([pose_3d[17][0], pose_3d[18][0]], [pose_3d[17][1], pose_3d[18][1]], zs=[pose_3d[17][2], pose_3d[18][2]],
                linewidth=3, color=b)
        ax.plot([pose_3d[18][0], pose_3d[19][0]], [pose_3d[18][1], pose_3d[19][1]], zs=[pose_3d[18][2], pose_3d[19][2]],
                linewidth=3, color=b)
        ax.plot([pose_3d[19][0], pose_3d[20][0]], [pose_3d[19][1], pose_3d[20][1]], zs=[pose_3d[19][2], pose_3d[20][2]],
                linewidth=3, color=b)
        return ax

class EgoDexLoss(nn.Module):
    def __init__(self, args, pretrain=False):
        super(EgoDexLoss, self).__init__()
        # Initialize Parameters
        self.args = args
        self.pretrain = pretrain
        self.alpha_2d = 5.
        self.alpha_3d = 100 #100
        self.alpha_mask = 0. # 100
        self.alpha_reg = 0#10
        self.alpha_beta = 0#10000
        self.alpha_camera = 1
        self.n_meshes = 778
        self.img_size = 224

    def getRampUpScale(self, epoch):
        if self.pretrain:
            return torch.ones(1).cuda()
        else:
            return torch.ones(1).cuda()
            # return torch.FloatTensor([(epoch+1) / self.args.max_epochs_ramp_up]).cuda()

    def forward(self, epoch, mask, predictions, targets):
        # Initialize predictions
        x2d_pred, x3d_pred, camera_param_pred, theta, beta = predictions
        # Initialize targets
        joint_2d_target, joint_3d_target, verts_3d_target, camera_param_target, dataset_type = targets
        # Initialize Variables
        batch_size, seq_size, _ = x2d_pred.size()
        # Get Vectors
        joint_2d_pred = torch.stack((x2d_pred[:, :, :42:2], x2d_pred[:, :, 1:42:2]), dim=3)  # x_hat
        y_hat = x2d_pred[:, :, 42:].view(batch_size, seq_size, 778, 2)
        joint_3d_pred = torch.stack((x3d_pred[:, :, :63:3], x3d_pred[:, :, 1:63:3], x3d_pred[:, :, 2:63:3]), dim=3)
        verts_3d_pred = torch.stack((x3d_pred[:, :, 63::3], x3d_pred[:, :, 64::3], x3d_pred[:, :, 65::3]), dim=3)

        # Calculate the Losses - 2D joint re-projection loss
        loss_2d = torch.abs((joint_2d_pred[:, -1, [4, 8, 12, 16, 20], :].view(batch_size, -1) / self.img_size
                             - joint_2d_target[:, -1, :, :].view(batch_size, -1) / self.img_size)).sum(1).mean()
        loss_2d = self.alpha_2d * loss_2d

        # Calculate the Losses - Temporal loss
        loss_temp = torch.zeros(1).cuda()
        # Calculate the Losses - Hand mask loss
        loss_mask = torch.zeros(1).cuda()
        # Calculate the Losses - Camera loss
        loss_camera = torch.zeros(1).cuda()
        # Calculate the Losses - Regularization loss
        loss_reg = torch.zeros(1).cuda()
        # Calculate the Losses - 3D joint loss (Only the STEREO dataset)
        diff_3d = torch.zeros(batch_size, 1, 5, 3).cuda()
        # last frame only
        joint3d_pred = joint_3d_pred[:, -1, [4, 8, 12, 16, 20], :]  # last frame
        joint3d_targ = joint_3d_target[:, -1, :, :]
        # joint2d_pred = joint_2d_pred[:, -1, [4, 8, 12, 16, 20], :]  # last frame
        # joint2d_targ = joint_2d_target[:, -1, :, :]
        #loop over batch
        for b in range(batch_size):
            targs3d = joint3d_targ[b, :]  # 5, 3
            preds3d = joint3d_pred[b, :]
            preds3d[joint3d_targ[b, :] == 0.] = 0.
            visible_indice = (joint3d_targ[b, :] != 0.).nonzero()
            visible_indice_mask = joint3d_targ[b, :] != 0.
            ## normalize
            preds3d, pred_min, pred_max = self.normalize_joints_scale(preds3d.clone())
            targs3d, targ_min, targ_max = self.normalize_joints_scale(targs3d.clone())

            maxp, _ = torch.max(preds3d.clone(), dim=0, keepdim=True)
            maxt, _ = torch.max(targs3d.clone(), dim=0, keepdim=True)
            preds3d = self.center_joints_scale(preds3d.clone(), maxp)
            targs3d = self.center_joints_scale(targs3d.clone(), maxt)
            # plt.figure()
            # ax = plt.axes(projection='3d')
            # ax.scatter3D(xs=preds3d[:, 0].clone().detach().cpu().numpy(),
            #             ys=preds3d[:, 1].clone().detach().cpu().numpy(),
            #             zs=preds3d[:, 2].clone().detach().cpu().numpy(),
            #             c='blue')
            # ax.scatter3D(xs=targs3d[:, 0].clone().detach().cpu().numpy(),
            #             ys=targs3d[:, 1].clone().detach().cpu().numpy(),
            #             zs=targs3d[:, 2].clone().detach().cpu().numpy(),
            #             c='red')
            # plt.show()
            targs3d = targs3d[visible_indice_mask].view(torch.unique(visible_indice[:, 0]).size()[0], 3)  # joint, coord
            preds3d = preds3d[visible_indice_mask].view(torch.unique(visible_indice[:, 0]).size()[0], 3)  # joint, coord
            diff_3d_ego = (targs3d - preds3d) * (pred_max - pred_min).repeat(torch.unique(visible_indice[:, 0]).size()[0],1)
            diff_3d[b, 0, visible_indice_mask] = diff_3d_ego.view(torch.unique(visible_indice[:, 0]).size()[0] * 3)

        loss_3d = self.alpha_3d * torch.pow(diff_3d.view(batch_size, -1), 2).sum(1).mean().cuda()
        # Weighted sum
        loss = loss_2d + loss_3d + loss_mask + loss_reg + loss_camera + loss_temp

        # Initialize Average Distance Storage
        avg_distance_2d = list()
        avg_distance_3d = list()
        for _ in range(5):
            avg_distance_2d.append(None)
            avg_distance_3d.append(None)
        # Calculate euclidean distance
        diff_2d = joint_2d_pred[:, -1, [4, 8, 12, 16, 20], :].view(batch_size, -1, 2) - joint_2d_target[:, -1, :, :].view(batch_size, -1, 2)
        euclidean_dist_2d = np.sqrt(np.sum(np.square(diff_2d.detach().cpu().numpy()), axis=2))
        euclidean_dist_3d = np.sqrt(np.sum(np.square(diff_3d.squeeze(1).detach().cpu().numpy()), axis=2)) * 0.7
        for i in range(5):
            avg_distance_2d[i] = euclidean_dist_2d[:, i]
            avg_distance_3d[i] = euclidean_dist_3d[:, i]
        return loss, [loss_2d.item(), loss_3d.item(), loss_mask.item(), loss_reg.item(), loss_camera.item(), avg_distance_2d, avg_distance_3d]

    def normalize_joints_scale(self, hand_joints):
        min_joints, _ = torch.min(hand_joints, dim=0, keepdim=True)
        max_joints, _ = torch.max(hand_joints, dim=0, keepdim=True)
        hand_joints[:, 0] = (hand_joints[:, 0] - min_joints[:, 0]) / (max_joints[:, 0] - min_joints[:, 0])
        hand_joints[:, 1] = (hand_joints[:, 1] - min_joints[:, 1]) / (max_joints[:, 0] - min_joints[:, 0])
        hand_joints[:, 2] = (hand_joints[:, 2] - min_joints[:, 2]) / (max_joints[:, 0] - min_joints[:, 0])
        return hand_joints, min_joints, max_joints

    def center_joints_scale(self, hand_joints, max_joints):
        hand_joints[:, 0] = hand_joints[:, 0] - max_joints[:, 0]
        hand_joints[:, 1] = hand_joints[:, 1] - max_joints[:, 1]
        hand_joints[:, 2] = hand_joints[:, 2] - max_joints[:, 2]
        return hand_joints




class DexterObjLoss(nn.Module):
    def __init__(self, args, pretrain=False):
        super(DexterObjLoss, self).__init__()
        # Initialize Parameters
        self.args = args
        self.pretrain = pretrain
        self.alpha_2d = 1
        self.alpha_3d = 100 #100
        self.alpha_mask = 0. # 100
        self.alpha_reg = 0#10
        self.alpha_beta = 0#10000
        self.alpha_camera = 1
        self.n_meshes = 778
        self.img_size = 224

    def getRampUpScale(self, epoch):
        if self.pretrain:
            return torch.ones(1).cuda()
        else:
            return torch.ones(1).cuda()
            # return torch.FloatTensor([(epoch+1) / self.args.max_epochs_ramp_up]).cuda()

    def forward(self, epoch, mask, predictions, targets):
        # Initialize predictions
        x2d_pred, x3d_pred, camera_param_pred, theta, beta = predictions
        # Initialize targets
        joint_2d_target, joint_3d_target, verts_3d_target, camera_param_target, dataset_type = targets
        # Initialize Variables
        batch_size, seq_size, _ = x2d_pred.size()
        # Get Vectors
        joint_2d_pred = torch.stack((x2d_pred[:, :, :42:2], x2d_pred[:, :, 1:42:2]), dim=3)  # x_hat
        y_hat = x2d_pred[:, :, 42:].view(batch_size, seq_size, 778, 2)
        joint_3d_pred = torch.stack((x3d_pred[:, :, :63:3], x3d_pred[:, :, 1:63:3], x3d_pred[:, :, 2:63:3]), dim=3)
        verts_3d_pred = torch.stack((x3d_pred[:, :, 63::3], x3d_pred[:, :, 64::3], x3d_pred[:, :, 65::3]), dim=3)

        # Calculate the Losses - 2D joint re-projection loss
        loss_2d = torch.abs((joint_2d_pred[:, -1, [4, 8, 12, 16, 20], :].view(batch_size, -1) / self.img_size
                             - joint_2d_target[:, -1, :, :].view(batch_size, -1) / self.img_size)).sum(1).mean()
        loss_2d = self.alpha_2d * loss_2d

        # Calculate the Losses - Temporal loss
        loss_temp = torch.zeros(1).cuda()
        # Calculate the Losses - Hand mask loss
        loss_mask = torch.zeros(1).cuda()
        # Calculate the Losses - Camera loss
        loss_camera = torch.zeros(1).cuda()
        # Calculate the Losses - Regularization loss
        loss_reg = torch.zeros(1).cuda()
        # Calculate the Losses - 3D joint loss (Only the STEREO dataset)
        diff_3d = torch.zeros(batch_size, 1, 5, 3).cuda()
        # last frame only
        joint3d_pred = joint_3d_pred[:, -1, [4, 8, 12, 16, 20], :]  # last frame
        joint3d_targ = joint_3d_target[:, -1, :, :]
        # joint2d_pred = joint_2d_pred[:, -1, [4, 8, 12, 16, 20], :]  # last frame
        # joint2d_targ = joint_2d_target[:, -1, :, :]
        #loop over batch
        for b in range(batch_size):
            targs3d = joint3d_targ[b, :]  # 5, 3
            preds3d = joint3d_pred[b, :]
            targs3d[joint3d_targ[b, :, 2] == 32001, :] = 0.
            preds3d[joint3d_targ[b, :, 2] == 32001, :] = 0.
            visible_indice = (joint3d_targ[b, :, 2] != 32001).nonzero()
            visible_indice_mask = joint3d_targ[b, :, 2] != 32001
            ## normalize
            preds3d, pred_min, pred_max = self.normalize_joints_scale(preds3d.clone())
            targs3d, targ_min, targ_max = self.normalize_joints_scale(targs3d.clone())

            maxp, _ = torch.max(preds3d.clone(), dim=0, keepdim=True)
            maxt, _ = torch.max(targs3d.clone(), dim=0, keepdim=True)
            preds3d = self.center_joints_scale(preds3d.clone(), maxp)
            targs3d = self.center_joints_scale(targs3d.clone(), maxt)
            targs3d = targs3d[visible_indice_mask, :]
            preds3d = preds3d[visible_indice_mask, :]
            # plt.figure()
            # ax = plt.axes(projection='3d')
            # ax.scatter3D(xs=preds3d[:, 0].clone().detach().cpu().numpy(),
            #             ys=preds3d[:, 1].clone().detach().cpu().numpy(),
            #             zs=preds3d[:, 2].clone().detach().cpu().numpy(),
            #             c='blue')
            # ax.scatter3D(xs=targs3d[:, 0].clone().detach().cpu().numpy(),
            #             ys=targs3d[:, 1].clone().detach().cpu().numpy(),
            #             zs=targs3d[:, 2].clone().detach().cpu().numpy(),
            #             c='red')
            # plt.show()
            targs3d = targs3d[visible_indice_mask].view(torch.unique(visible_indice[:, 0]).size()[0], 3)  # joint, coord
            preds3d = preds3d[visible_indice_mask].view(torch.unique(visible_indice[:, 0]).size()[0], 3)  # joint, coord
            diff_3d_dex = (targs3d - preds3d) * (pred_max - pred_min).repeat(torch.unique(visible_indice[:, 0]).size()[0],1)
            diff_3d[b, 0, visible_indice_mask, :] = diff_3d_dex.view(visible_indice.size()[0], 3)
        loss_3d = self.alpha_3d * torch.pow(diff_3d.view(batch_size, -1), 2).sum(1).mean().cuda()
        # Weighted sum
        loss = loss_2d + loss_3d + loss_mask + loss_reg + loss_camera + loss_temp

        # Initialize Average Distance Storage
        avg_distance_2d = list()
        avg_distance_3d = list()
        for _ in range(5):
            avg_distance_2d.append(None)
            avg_distance_3d.append(None)
        # Calculate euclidean distance
        diff_2d = joint_2d_pred[:, -1, [4, 8, 12, 16, 20], :].view(batch_size, -1, 2)\
                  - joint_2d_target[:, -1, :, :].view(batch_size, -1, 2)
        euclidean_dist_2d = np.sqrt(np.sum(np.square(diff_2d.detach().cpu().numpy()), axis=2))
        euclidean_dist_3d = np.sqrt(np.sum(np.square(diff_3d.squeeze(1).detach().cpu().numpy()), axis=2))
        for i in range(5):
            avg_distance_2d[i] = euclidean_dist_2d[:, i]
            avg_distance_3d[i] = euclidean_dist_3d[:, i]
        return loss, [loss_2d.item(), loss_3d.item(), loss_mask.item(), loss_reg.item(), loss_camera.item(), avg_distance_2d, avg_distance_3d]

    def normalize_joints_scale(self, hand_joints):
        min_joints, _ = torch.min(hand_joints, dim=0, keepdim=True)
        max_joints, _ = torch.max(hand_joints, dim=0, keepdim=True)
        hand_joints[:, 0] = (hand_joints[:, 0] - min_joints[:, 0]) / (max_joints[:, 0] - min_joints[:, 0])
        hand_joints[:, 1] = (hand_joints[:, 1] - min_joints[:, 1]) / (max_joints[:, 0] - min_joints[:, 0])
        hand_joints[:, 2] = (hand_joints[:, 2] - min_joints[:, 2]) / (max_joints[:, 0] - min_joints[:, 0])
        return hand_joints, min_joints, max_joints

    def center_joints_scale(self, hand_joints, max_joints):
        hand_joints[:, 0] = hand_joints[:, 0] - max_joints[:, 0]
        hand_joints[:, 1] = hand_joints[:, 1] - max_joints[:, 1]
        hand_joints[:, 2] = hand_joints[:, 2] - max_joints[:, 2]
        return hand_joints