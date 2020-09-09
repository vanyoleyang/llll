import os
import sys
import torch
import torchvision.transforms as transforms
import random
from args import get_parser
from dataset import HAND3D_seq, STEREO_seq, RWORLD_seq, EgoDexter_seq, DexterObject_seq, REAL_seq, REAL_seq3, FHAD
from model_encoder import EncoderBase
from model_encoder_grey import EncoderBase as EncoderBaseGrey
from model_LSTM import EncoderLSTM
from model_convLSTM import EncoderConvLSTM
from model_convLSTM_stoch import EncoderConvLSTM_stoch
from model_I3D import I3D
from model_MFNet import MotionFeatureNet
from loss import FHADLoss#, Hand3DLoss, STBLoss, RWorldLoss, DexterObjLoss, EgoDexLoss, Hand3DLoss_wKLD
from trainer import pretrain, train, valid
from utils import loadCheckpoint


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

if __name__ == "__main__":
    # Initialize variables
    parser = get_parser()
    args = parser.parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = True
    print('CUDA Version: %s/10.0.130' % torch.version.cuda)
    print('CUDNN Version: %s/7600' % torch.backends.cudnn.version())
    print('PyTorch Version: %s/1.2.0\n' % torch.__version__)
    args.multi_gpu = False
    if not args.multi_gpu:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    args.device = torch.device('cuda')
    print(args)

    # Initialize dataloaders

    transform_train = transforms.Compose([transforms.Lambda(lambda images: [transforms.Resize([224, 224])(image) for image in images]),
                                          transforms.Lambda(lambda images: torch.stack([transforms.ToTensor()(image) for image in images])),
                                          transforms.Lambda(lambda images: torch.stack([transforms.Normalize([0, 0, 0], [1, 1, 1])(image) for image in images]))])
    transform_valid = transforms.Compose([transforms.Lambda(lambda images: [transforms.Resize([224, 224])(image) for image in images]),
                                          transforms.Lambda(lambda images: torch.stack([transforms.ToTensor()(image) for image in images])),
                                          transforms.Lambda(lambda images: torch.stack([transforms.Normalize([0, 0, 0], [1, 1, 1])(image) for image in images]))])

    # SeqHAND
    # dataset_HAND3D_train = HAND3D_seq(args, mode='train', transform=transform_train)
    dataset_train = FHAD('/media/vanyole/VANYOLE/HPE/Dataset/FHAD', mode='train')
    dataset_FHAD_loader = torch.utils.data.DataLoader(dataset=dataset_train, batch_size=args.batch_size_train,
                                                      shuffle=True, sampler=None, num_workers=12, pin_memory=True, drop_last=True)
    dataset_valid = FHAD('/media/vanyole/VANYOLE/HPE/Dataset/FHAD', mode='valid')
    dataset_FHAD_loader_valid = torch.utils.data.DataLoader(dataset=dataset_valid, batch_size=args.batch_size_train,
                                                            shuffle=True, sampler=None, num_workers=12, pin_memory=True,
                                                            drop_last=False)
    model = EncoderConvLSTM(args, 0, latent_dim=args.latent_dim, hidden_dim=args.hidden_dim,
                            lstm_layers=args.n_lstm_layers,
                            attention=args.attention, n_classes=args.n_classes,
                            backbone=args.backbone, pretrain=args.pretrained_backbone)

    # Set CUDA
    if args.multi_gpu:
        model = torch.nn.DataParallel(model, device_ids=range(args.n_gpus)).to('cuda')
    else:
        model = model.to(device=args.device)


    # args.lr_base_train = 2e-8
    # Initialize Optimizer
    optimizer = torch.optim.Adam(model.parameters(), args.lr_base_pretrain, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-5)

    #print(args)
    print('model param # :: ', count_parameters(model))

    # Fix parameters
    # for name, param in model.named_parameters():
    #     if param.requires_grad:
    #         if name.split('.')[1] == 'conv_lstm':
    #             param.requires_grad = False
    #             print(name, ' fixed')

    FHAD_criterion = FHADLoss(args).to(device=args.device)
    dataloaders = {'HAND3D': {'train': dataset_FHAD_loader, 'valid': dataset_FHAD_loader_valid}}
    args.n_kps = 21
    loss = FHAD_criterion
    args.model_id = 'pretrain_best'
    _, model, optimizer = loadCheckpoint(args, model, optimizer, best=True, load_pretrain=False)
    args.model_id += '_FHAD'

    _ = valid(args, -1, args.max_epochs_train, 0, dataset_FHAD_loader_valid, model, loss, display_2D=False, display_3D=False)
    # Train
    train(args, dataloaders, model, loss, optimizer)

    ## TODO : RL loss and variational inference for uncertainties

