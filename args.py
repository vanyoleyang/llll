import argparse
from datetime import datetime

def get_parser():
    parser = argparse.ArgumentParser(description='HAND3D_SEQ')

    parser.add_argument('--root', type=str,
                        default='/home/vanyole/Research/HPE/Synth_hand_generation/data/', help='')
    # parser.add_argument('--root_stereo', type=str, default='/media/labpc/VANYOLE/Dataset/STB_cropped_right/', help='')
    parser.add_argument('--root_stereo', type=str, default='/home/vanyole/Research/HPE/Dataset/STB_cropped_right/', help='')
    parser.add_argument('--root_REAL', type=str, default='/home/labpc/HandPoseEstimation_John_SNU/Real_data/', help='')
    parser.add_argument('--root_real', type=str, default='/media/vanyole/VANYOLE/HPE/Dataset/rworld_data/rworld_data', help='')
    parser.add_argument('--root_egodexter', type=str, default='/home/vanyole/Research/HPE/Dataset/EgoDexter_seq', help='')
    parser.add_argument('--root_dexterobject', type=str, default='/home/vanyole/Research/HPE/Dataset/DexterObject_seq', help='')
    parser.add_argument('--root_mano', type=str,
                        default='/media/vanyole/VANYOLE/HPE/manopth-master/mano/models',
                        help='')
    parser.add_argument('--seed', type=int, default=100, help='random Seed')
    parser.add_argument('--n_gpus', type=int, default=1, help='number of GPUs')

    parser.add_argument('--max_epochs_pretrain', type=int, default=100, help='number of maximum epochs')
    parser.add_argument('--max_epochs_train', type=int, default=140, help='number of maximum epochs')
    parser.add_argument('--max_epochs_ramp_up', type=int, default=50, help='number of maximum epochs')

    parser.add_argument('--lr_policy_pretrain', type=str, default='multistep', help='Learning Rate Policy')
    parser.add_argument('--lr_policy_train', type=str, default='multistep', help='Learning Rate Policy')
    parser.add_argument('--lr_policy_param_pretrain', type=dict, default={'stepvalue': [], 'gamma': 0.1}, help='Learning Rate Policy')
    parser.add_argument('--lr_policy_param_train', type=dict, default={'stepvalue': [60], 'gamma': 0.5}, help='Learning Rate Policy') #[20, 80]
    parser.add_argument('--lr_base_pretrain', type=float, default=0.00001, help='initial learning rate for learning camera parameters')
    parser.add_argument('--lr_base_train', type=float, default=0.0001, help='initial learning rate for training')
    parser.add_argument('--weight_decay', type=float, default=0.00001, help='weight decay')

    parser.add_argument('--batch_size_train', type=int, default=24, help='number of training samples')
    parser.add_argument('--batch_size_valid', type=int, default=24, help='number of training samples')
    parser.add_argument('--seq_size_train', type=int, default=7, help='number of frames in each sequence')
    parser.add_argument('--seq_size_valid', type=int, default=-1, help='number of frames in each sequence')
    parser.add_argument('--channel_size', type=int, default=3, help='channel Size: number of channels')
    parser.add_argument('--img_size', type=int, default=224, help='image size: height / width dimension')
    parser.add_argument('--latent_dim', type=int, default=512, help='dimension of the latent variables, The number of expected features in the input x of LSTM layer')
    parser.add_argument('--hidden_dim', type=int, default=1024, help='The number of features in the hidden state h of LSTM layer')
    parser.add_argument('--n_lstm_layers', type=int, default=1, help='Number of LSTM layers')
    parser.add_argument('--bidirectional', type=bool, default=True, help='If True, becomes a bidirectional LSTM')
    parser.add_argument('--attention', type=bool, default=False, help='If True, attention mechanism is applied')
    parser.add_argument('--n_classes', type=int, default=26, help='Number of classes')
    parser.add_argument('--n_kps', type=int, default=21, help='Number of Joints')
    parser.add_argument('--backbone', type=str, default='resnet50', help='Backbone: resnet50 | resnet101 | resnet152')
    parser.add_argument('--pretrained_backbone', type=bool, default=False, help='Pretrain')

    parser.add_argument('--interval_checkpoint', type=int, default=10, help='Checkpoint')
    parser.add_argument('--interval_display_train', type=int, default=100, help='Interval')
    parser.add_argument('--interval_display_valid', type=int, default=999, help='Interval')

    parser.add_argument('--model_name', type=str, default='EncoderConvLSTM', help='[ Encoder | EncoderConvLSTM_stoch | EncoderLSTM | EncoderConvLSTM | EncoderConvLSTM_RN | MFNet | I3D ]')
    parser.add_argument('--model_id', type=str, default=datetime.today().strftime('%Y%m%d_%H%M%S'), help='model_id')
    parser.add_argument('--valid_best', type=float, default=0.0, help='Delta')
    parser.add_argument('--valid_delta', type=float, default=0.0, help='Delta')

    return parser


if __name__ == "__main__":
    parser = get_parser()
    args_dict = parser.parse_args()

# EncoderLSTM
# DIST: 4.27px( 3.68px)  8.84mm( 7.85mm) AUC: 0.858(0.985)
# EncoderConvLSTM
