import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
import os
import functools
import pickle
import json
import numpy as np
import scipy.io
import cv2
import matplotlib.pyplot as plt
import natsort
import imageio

extensions = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.gif']
STEREO_TRAIN = ['B1Counting', 'B1Random', 'B2Counting', 'B2Random', 'B3Counting', 'B3Random', 'B4Counting', 'B4Random', 'B5Counting', 'B5Random']
STEREO_VALID = ['B6Counting', 'B6Random']
STEREO_TRAIN_LABEL = ['B1Counting_BB.mat', 'B1Random_BB.mat', 'B2Counting_BB.mat', 'B2Random_BB.mat', 'B3Counting_BB.mat', 'B3Random_BB.mat', 'B4Counting_BB.mat', 'B4Random_BB.mat', 'B5Counting_BB.mat', 'B5Random_BB.mat']
STEREO_VALID_LABEL = ['B6Counting_BB.mat', 'B6Random_BB.mat']

def flip(x, dim):
    indices = [slice(None)] * x.dim()
    indices[dim] = torch.arange(x.size(dim) - 1, -1, -1, dtype=torch.long, device=x.device)
    return x[tuple(indices)]


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        if path.split('.')[-1] == 'gif':
            img = imageio.mimread(f)
            img = [Image.fromarray(img[idx]).convert('RGB') for idx in range(len(img))]
        else:
            img = Image.open(f).convert('RGB')
    return img

def add_margin(pil_img, top=40, right=40, bottom=40, left=40, color='black'):
    width, height = pil_img.size
    new_width = width + right + left
    new_height = height + top + bottom
    result = Image.new(pil_img.mode, (new_width, new_height), color)
    result.paste(pil_img, (left, top))
    return result


def pil_loader_rworld(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        if path.split('.')[-1] == 'gif':
            img = imageio.mimread(f)
            img = [add_margin(Image.fromarray(img[idx]).convert('RGB')) for idx in range(len(img))]
        else:
            img = Image.open(f).convert('RGB')
    return img

def accimage_loader(path):
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def default_loader(path):
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)


def has_file_allowed_extension(filename, extensions):
    filename_lower = filename.lower()
    return any(filename_lower.endswith(ext) for ext in extensions)


def make_dataset(dir, extensions):
    images = []
    dir = os.path.expanduser(dir)
    for root, _, fnames in sorted(os.walk(dir)):
        fnames = natsort.natsorted(fnames)
        for fname in fnames:
            if has_file_allowed_extension(fname, extensions):
                path = os.path.join(root, fname)
                images.append(path)
    return images



class FHAD(data.Dataset):
    def __init__(self, root_FHAD, mode='train', transform=None, loader=default_loader, seq_size=10):
        import collections, random
        self.seq_size = seq_size
        self.root = root_FHAD
        self.mode = mode
        self.loader = loader
        self.transform = transforms.Compose([transforms.Lambda(lambda images: [transforms.Resize([224, 224])(image) for image in images]),
                                          transforms.Lambda(lambda images: torch.stack([transforms.ToTensor()(image) for image in images])),
                                          transforms.Lambda(lambda images: torch.stack([transforms.Normalize([0, 0, 0], [1, 1, 1])(image) for image in images]))])

        self.FHAD2MANO = [0,1,6,7,8,2,9,10,11,3,12,13,14,4,15,16,17,5,18,19,20]
        # self.generate_dataset()
        self.path_image = make_dataset(os.path.join(self.root, 'cross_sub', self.mode, 'images'), extensions)
        gts = pickle.load(open(os.path.join(self.root, 'cross_sub', self.mode, 'gt_w2Dcrop.pickle'), 'rb'), encoding='latin1')
        self.targ_2d_seq = gts['2ds']
        self.targ_3d_seq = gts['3ds']
        if self.mode == 'valid':
            rand_indices = random.choices(range(len(self.path_image)), k = 5000)
            new_path_img, new_targ_2d, new_targ_3d = [], [], []
            for index_ in rand_indices :
                new_path_img.append(self.path_image[index_])
                new_targ_2d.append(self.targ_2d_seq[index_])
                new_targ_3d.append(self.targ_3d_seq[index_])
            self.path_image = new_path_img
            self.targ_2d_seq = new_targ_2d
            self.targ_3d_seq = new_targ_3d
        #
        # # test_ 2D cropped joint locations
        # new_coord, bbox_info = self.crop_around_hand(self.targ_2d_seq[0][0])
        # self.diplay_2d(self.path_img_seq[0][0], new_coord, bbox_info=bbox_info)
        # self.diplay_2d(self.path_img_seq[0][0], self.targ_2d_seq[0][0])

    def generate_dataset(self):
        print('\n Generating %s Dataset....... \n' % self.mode)
        self.path_img_seq = []
        self.targ_3d_seq = []
        self.targ_2d_seq = []
        if self.mode == 'valid' :
            # subjects = [subjects_[subcross][1]]
            subjects = [2, 5, 6]
        elif self.mode == 'train' :
            # subjects = list(subjects_[subcross][0])
            subjects = [1, 3, 4]
        else :
            raise Exception("mode can be either valid(test), train")
        for subj_ind in subjects:
            subj_direc = 'Subject_' + str(subj_ind)
            action_direc_img = os.path.join(self.root, 'Video_files', subj_direc)
            action_direc_ano = os.path.join(self.root, 'Hand_pose_annotation_v1', subj_direc)
            for action in os.listdir(action_direc_img):
                # action_direc + action_ ("/write/") +
                case_direc_img = os.path.join(action_direc_img, action)
                case_direc_ano = os.path.join(action_direc_ano, action)
                for case in os.listdir(case_direc_img):
                    frame_n = 0
                    # action_direc + action_ ("/write/") + case ("/2/")
                    image_direc = os.path.join(case_direc_img, case, 'color')
                    annot_direc = os.path.join(case_direc_ano, case)
                    annots = np.loadtxt(annot_direc + '/skeleton.txt')[:, 1:]
                    for index, image_path in enumerate(natsort.natsorted(os.listdir(image_direc))):
                        # self.path_image_seq.append(image_direc + image_path)
                        annots_3d = annots[index].reshape((21, 3))
                        annots_2d = self.get_2d_coord(annots_3d)
                        if frame_n == 0:
                            img_sequence = collections.deque([os.path.join(image_direc, image_path)] * self.seq_size,
                                                             maxlen=self.seq_size)
                            anot_sequence = collections.deque([annots_3d] * self.seq_size,
                                                              maxlen=self.seq_size)
                            anot_2d_seq = collections.deque([annots_2d] * self.seq_size,
                                                            maxlen=self.seq_size)
                        else:
                            img_sequence.append(os.path.join(image_direc, image_path))
                            anot_sequence.append(annots_3d)
                            anot_2d_seq.append(annots_2d)
                        self.path_img_seq.append(list(img_sequence))
                        self.targ_3d_seq.append(np.stack(anot_sequence, 0))
                        self.targ_2d_seq.append(np.stack(anot_2d_seq, 0))
                        frame_n += 1
        #
        # self.new_coord, bbox_info = self.crop_around_hand(self.targ_2d_seq[50000][0])
        # self.diplay_2d(self.path_img_seq[50000][0], self.new_coord, bbox_info=bbox_info)
        self.new_img_direc = os.path.join(self.root, 'cross_sub', self.mode, 'images')

        self.targ_2d_seq_crop = []
        for seq_ind, image_paths in enumerate(self.path_img_seq):
            print(seq_ind, ' / ', len(self.path_img_seq))
            # print(image_paths)
            images = []
            targ_2d = []
            for img_path_ind, image_path in enumerate(image_paths):
                image = self.loader(image_path)
                joint_2d_crop, (x, y, w, l) = self.crop_around_hand(self.targ_2d_seq[seq_ind][img_path_ind])
                joint_2d_crop[:, 1] = joint_2d_crop[:, 1] * 224 / w
                joint_2d_crop[:, 0] = joint_2d_crop[:, 0] * 224 / w
                targ_2d.append(joint_2d_crop)
                image = np.array(image.crop((y, x, y + l, x + w)).resize((224, 224)))
                images.append(image)
            self.targ_2d_seq_crop.append(np.stack(targ_2d, 0))
            imageio.mimsave(self.new_img_direc + '/%08d.gif' % seq_ind, images)
        targets = {'3ds' : self.targ_3d_seq, '2ds' : self.targ_2d_seq_crop}
        with open(os.path.join(self.root, 'cross_sub', self.mode, 'gt_w2Dcrop.pickle'), 'wb') as f:
            pickle.dump(targets, f)



    def get_2d_coord(self, coord_3d):
        u0, v0, fx, fy = 935.732544, 540.681030, 1395.749023, 1395.749268
        self.cam_int = np.array([[fx, 0., u0], [0., fy, v0], [0., 0., 1.]])
        self.cam_ext = np.array([[0.999988496304, -0.00468848412856, 0.000982563360594, 25.7],
                                 [0.00469115935266, 0.999985218048, -0.00273845880292, 1.22],
                                 [-0.000969709653873, 0.00274303671904, 0.99999576807, 3.902]])

        ll = np.concatenate([coord_3d, np.ones((21, 1))], axis=1)
        d3 = np.matmul(self.cam_ext, ll.T)
        d2 = np.matmul(self.cam_int, d3)
        d2[0, :] /= d2[2, :]
        d2[1, :] /= d2[2, :]
        return d2[:2, :].T

    def diplay_2d(self, img_pth, coord_2d, bbox_info = None):
        from PIL import Image, ImageDraw
        im = self.loader(img_pth)

        if bbox_info != None  :
            x, y, w, l = bbox_info
            im = im.crop((y, x, y+l, x+w))
            coord_2d = coord_2d * 224 / w
            print(coord_2d)
        else :
            w, l = 1920, 1080
            # print(coord_2d)
            coord_2d[:, 1] = coord_2d[:, 1] * 224 / l
            coord_2d[:, 0] = coord_2d[:, 0] * 224 / w

        im = im.resize((224, 224))

        draw = ImageDraw.Draw(im)
        draw.point(coord_2d.reshape(42,).tolist(), fill='white')
        im.save('2d_plot', "JPEG")


    def __getitem__(self, index):
        joint_2d_seq_crop = self.targ_2d_seq[index]
        joint_3d_seq = self.targ_3d_seq[index]
        image_paths = self.path_image[index]
        images = self.loader(image_paths)

        # reorder MANO
        joint_2d_seq = torch.tensor(joint_2d_seq_crop)[:, self.FHAD2MANO, :]
        joint_3d_seq = torch.tensor(joint_3d_seq)[:, self.FHAD2MANO, :]
        images = self.transform(images)

        # unused data
        mask = torch.zeros(1)
        verts_3d = torch.zeros(1)
        cam_param = torch.zeros(1)
        dataset_type = torch.zeros(1)
        return images, mask, joint_2d_seq, joint_3d_seq, verts_3d, cam_param, dataset_type, index

    def crop_around_hand(self, coords_2d, scale=2.4):
        maxx, minx = np.max(coords_2d[:, 0]), np.min(coords_2d[:, 0])
        maxy, miny = np.max(coords_2d[:, 1]), np.min(coords_2d[:, 1])
        w_h, l_h = maxx - minx, maxy - miny
        ll = scale * max(w_h, l_h)
        hand_cent = coords_2d.mean(0)

        bbox_x = hand_cent[1] - ll / 2.
        bbox_y = hand_cent[0] - ll / 2.
        bbox_w, bbox_l = ll, ll
        coords_2d[:, 1] = (coords_2d[:, 1] - bbox_x) #/ ll
        coords_2d[:, 0] = (coords_2d[:, 0] - bbox_y) #/ ll
        return coords_2d, (int(bbox_x), int(bbox_y), int(bbox_w), int(bbox_l))


    def __len__(self):
        return len(self.path_image)


class STEREO(data.Dataset):
    def __init__(self, args, mode='train', transform=None, loader=default_loader):
        self.root = args.root
        self.folder = STEREO_TRAIN if mode == 'train' else STEREO_VALID
        self.file_label = STEREO_TRAIN_LABEL if mode == 'train' else STEREO_VALID_LABEL
        self.mode = mode
        self.loader = loader
        self.transform = transform
        self.path_image = []
        for folder in self.folder:
            self.path_image = self.path_image + make_dataset(os.path.join(self.root, folder), extensions)[:1500]
        if len(self.path_image) == 0:
            raise(RuntimeError("Found 0 files in subfolders of: " + os.path.join(self.root, 'new_data') + "\n"
                               "Supported extensions are: " + ",".join(extensions)))
        self.joint_3d_target = []
        for file in self.file_label:
            self.joint_3d_target.append(np.transpose(scipy.io.loadmat(os.path.join(self.root, 'labels', file))['handPara'], (2, 1, 0)))
        # To Tensor
        self.joint_3d_target = torch.from_numpy(np.concatenate(self.joint_3d_target, 0)).float()
        scale = torch.tensor([822.79041])/self.joint_3d_target[:, :, 2:]
        trans = torch.tensor([318.47345, 245.31296])
        self.joint_2d_target = self.joint_3d_target[:, :, :2] * scale + trans.unsqueeze(0).unsqueeze(1)  # [pixel]
        self.joint_2d_target = torch.cat((self.joint_2d_target, torch.ones(self.joint_2d_target.size(0), self.joint_2d_target.size(1), 1)), dim=2)
        self.n_total_sample = len(self.path_image)

    def __getitem__(self, index):
        path_image = self.path_image[index]
        image = self.loader(path_image)
        joint_2d_target = self.joint_2d_target[index]
        joint_3d_target = self.joint_3d_target[index]
        target_is_left = torch.tensor([1])
        target_2d_hand_box_center = joint_2d_target.mean(0)[:2]

        # Randomly Crop the Image
        # displayImage(index, image, joint_2d_target, target_is_left, target_2d_hand_box_center, '')
        length = 2.2*int((torch.max(joint_2d_target, 0)[0] - torch.min(joint_2d_target, 0)[0]).max())
        # random_scale = (2.2+(np.random.rand()-0.5))  # 1.7 - 2.7
        # length = int(length * random_scale)
        j = int(target_2d_hand_box_center[0].item()) - int(length/2) # + int((np.random.rand()-0.5)*(length/4))
        i = int(target_2d_hand_box_center[1].item()) - int(length/2) # + int((np.random.rand()-0.5)*(length/4))
        image = image.crop((j, i, j + length, i + length))

        # Scale & Translate the Target
        scale = self.transform.transforms[0].size
        joint_2d_target[:, 0] = (joint_2d_target[:, 0] - j) * (scale[0]/length)
        joint_2d_target[:, 1] = (joint_2d_target[:, 1] - i) * (scale[1]/length)
        target_2d_hand_box_center[0] = (target_2d_hand_box_center[0] - j) * (scale[0]/length)
        target_2d_hand_box_center[1] = (target_2d_hand_box_center[1] - i) * (scale[1]/length)

        # To Tensor
        if self.transform is not None:
            image = self.transform(image)
        joint_3d_target = to_mano(joint_3d_target)
        image = np.array(image)
        return image, joint_2d_target[:, :2], joint_3d_target, index


        # if self.transform is not None:
        #     image = self.transform(image)
        # # NEED TO FLIPLR (STEREO: Left Hand)
        # image = flip(image, dim=2)
        # joint_3d_target = flip_3d_coord(joint_3d_target)
        # joint_3d_target = torch.from_numpy(joint_3d_target).float().view(21, -1)
        # return image, joint_3d_target, index

    def __len__(self):
        return len(self.path_image)



class MPII_NZSL(data.Dataset):
    def __init__(self, root, mode='train', transform=None, loader=default_loader):
        self.root = root
        self.folder = 'manual_train' if mode == 'train' else 'manual_test'
        self.mode = mode
        self.loader = loader
        self.transform = transform
        self.path_image = make_dataset(os.path.join(self.root, self.folder), extensions)
        if len(self.path_image) == 0:
            raise(RuntimeError("Found 0 files in subfolders of: " + os.path.join(self.root, self.folder) + "\n"
                               "Supported extensions are: " + ",".join(extensions)))
        self.path_json = make_dataset(os.path.join(self.root, self.folder), ['.json'])
        if len(self.path_json) == 0:
            raise(RuntimeError("Found 0 files in subfolders of: " + os.path.join(self.root, self.folder) + "\n"
                               "Supported extensions are: " + ",".join(extensions)))
        data = []
        self.target = dict()
        self.joint_2d_target = []
        self.target_2d_hand_box_center = []
        self.target_is_left = []
        for idx in range(len(self.path_json)):
            data.append(json.load(open(self.path_json[idx], 'rb')))
        for key in data[906]:
            self.target[key] = []
            for idx in range(len(data)):
                self.target[key].append(data[idx][key])
        for idx in range(len(data)):
            self.joint_2d_target.append(np.array(self.target['hand_pts'][idx]))
            self.target_2d_hand_box_center.append(np.array(self.target['hand_box_center'][idx]))
            self.target_is_left.append(np.array(self.target['is_left'][idx]))
        # To Tensor
        self.joint_2d_target = torch.from_numpy(np.stack(self.joint_2d_target, 0)).float()
        self.target_2d_hand_box_center = torch.from_numpy(np.stack(self.target_2d_hand_box_center, 0)).float()
        self.target_is_left = torch.from_numpy(np.stack(self.target_is_left, 0)).bool()
        self.n_total_sample = len(self.path_image)

    def __getitem__(self, index):
        # Get a Sample
        path_image = self.path_image[index]
        image = self.loader(path_image)
        joint_2d_target = self.joint_2d_target[index]
        target_2d_hand_box_center = self.target_2d_hand_box_center[index]
        target_is_left = self.target_is_left[index]

        # Randomly Crop the Image
        # displayImage(index, image, joint_2d_target, target_is_left, target_2d_hand_box_center, '')
        length = (torch.max(joint_2d_target, 0)[0] - torch.min(joint_2d_target, 0)[0]).max()
        random_scale = (2.2+(np.random.rand()-0.5))  # 1.7 - 2.7
        length = int(length * random_scale)
        j = int(target_2d_hand_box_center[0].item()) - int(length/2) + int((np.random.rand()-0.5)*(length/4))
        i = int(target_2d_hand_box_center[1].item()) - int(length/2) + int((np.random.rand()-0.5)*(length/4))
        image = image.crop((j, i, j + length, i + length))

        # Scale & Translate the Target
        scale = self.transform.transforms[0].size
        joint_2d_target[:, 0] = (joint_2d_target[:, 0] - j) * (scale[0]/length)
        joint_2d_target[:, 1] = (joint_2d_target[:, 1] - i) * (scale[1]/length)
        target_2d_hand_box_center[0] = (target_2d_hand_box_center[0] - j) * (scale[0]/length)
        target_2d_hand_box_center[1] = (target_2d_hand_box_center[1] - i) * (scale[1]/length)

        # For the Right Hand Model
        if target_is_left.item():
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            joint_2d_target[:, 0] = scale[0] - joint_2d_target[:, 0]
            target_2d_hand_box_center[0] = scale[0] - target_2d_hand_box_center[0]

        # To Tensor
        if self.transform is not None:
            image = self.transform(image)

        # displayImage(index, image, joint_2d_target, target_is_left, target_2d_hand_box_center, 'cropped')
        return image, joint_2d_target, index


    def __len__(self):
        return len(self.path_image)


def displayImage(index, im, pts, is_left, center, name):
    plt.rcParams['figure.figsize'] = (20, 20)
    plt.rcParams['image.interpolation'] = 'nearest'
    plt.rcParams['image.cmap'] = 'gray'

    edges = [[0, 1], [1, 2], [2, 3], [3, 4], [0, 5], [5, 6], [6, 7], [7, 8], [0, 9], [9, 10], [10, 11], [11, 12], [0, 13], [13, 14], [14, 15], [15, 16], [0, 17], [17, 18], [18, 19], [19, 20]]

    # Each file contains 1 hand annotation, with 21 points in 'hand_pts' of size 21x3, following this scheme:
    # https://github.com/CMU-Perceptual-Computing-Lab/openpose/blob/master/doc/output.md#hand-output-format
    # The 3rd column is 1 if valid:
    pts = pts.numpy()
    center = center.numpy()
    invalid = pts[:, 2] != 1

    # Left hands are marked, but otherwise follow the same point order
    is_left = is_left.numpy()

    # Plot annotations
    plt.clf()
    plt.imshow(im)
    plt.plot(center[0], center[1], 'b*')
    for p in range(pts.shape[0]):
        if pts[p, 2] != 0:
            plt.plot(pts[p, 0], pts[p, 1], 'r.')
            plt.text(pts[p, 0], pts[p, 1], '{0}'.format(p))
    for ie, e in enumerate(edges):
        if np.all(pts[e, 2] != 0):
            rgb = matplotlib.colors.hsv_to_rgb([ie / float(len(edges)), 1.0, 1.0])
            plt.plot(pts[e, 0], pts[e, 1], color=rgb)
    if is_left:
        plt.text(10, 30, 'left', color='r', fontsize=24)
    plt.axis('off')
    plt.savefig('test_%08d_%s.jpg' %(index, name), bbox_inches='tight', dpi=96)

def to_mano(joint_3d_target):
    mean_bh_x, std_bh_x = 0.038911973892207914, 0.3864240027898126
    mean_bh_y, std_bh_y = - 0.31960639400535723, 0.4600456047757878
    mean_bh_z, std_bh_z = - 0.3012236227785493, 0.39613739471863946

    mean_mano_x, mean_mano_y, mean_mano_z = 0.084298, 0.003471684, 0.0054526534
    std_mano_x, std_mano_y, std_mano_z = 0.051625396, 0.063773718, 0.06369294

    mean_stb_x, std_stb_x = joint_3d_target[:, 0].mean(), joint_3d_target[:, 0].std()
    mean_stb_y, std_stb_y = joint_3d_target[:, 1].mean(), joint_3d_target[:, 1].std()
    mean_stb_z, std_stb_z = joint_3d_target[:, 2].mean(), joint_3d_target[:, 2].std()

    joint_3d_target[:, 0] = (joint_3d_target[:, 0] - mean_stb_x) * (std_mano_x / std_stb_x) + mean_mano_x
    joint_3d_target[:, 1] = (joint_3d_target[:, 1] - mean_stb_y) * (std_mano_y / std_stb_y) + mean_mano_y
    joint_3d_target[:, 2] = (joint_3d_target[:, 2] - mean_stb_z) * (std_mano_z / std_stb_z) + mean_mano_z
    return joint_3d_target

class PANOPTIC(data.Dataset):
    def __init__(self, root, mode='train', transform=None, loader=default_loader):
        self.root = root
        self.mode = mode
        self.loader = loader
        self.transform = transform
        self.path_image = make_dataset(os.path.join(self.root, 'imgs'), extensions)
        if len(self.path_image) == 0:
            raise(RuntimeError("Found 0 files in subfolders of: " + os.path.join(self.root, 'imgs') + "\n"
                               "Supported extensions are: " + ",".join(extensions)))
        self.target = dict()
        self.joint_2d_target = []
        self.target_2d_hand_box_center = []
        self.invalid = []
        data = json.load(open(os.path.join(self.root, 'hands_v143_14817.json'), 'rb'))
        for key in data['root'][0]:
            self.target[key] = []
            for idx in range(len(data['root'])):
                self.target[key].append(data['root'][idx][key])
        for idx in range(len(data['root'])):
            self.joint_2d_target.append(np.array(self.target['joint_self'][idx]))
            self.target_2d_hand_box_center.append(np.array(self.target['objpos'][idx]))
            self.invalid.append(self.joint_2d_target[idx][:, 2] != 1)
        # To Tensor
        self.joint_2d_target = torch.from_numpy(np.stack(self.joint_2d_target, 0)).float()
        self.target_2d_hand_box_center = torch.from_numpy(np.stack(self.target_2d_hand_box_center, 0)).float()
        self.n_total_sample = len(self.path_image)

    def __getitem__(self, index):
        # Get a Sample
        path_image = self.path_image[index]
        image = self.loader(path_image)
        joint_2d_target = self.joint_2d_target[index]
        target_2d_hand_box_center = self.target_2d_hand_box_center[index]

        # Randomly Crop the Image
        target_is_left = torch.tensor([0])
        # displayImage(index, image, joint_2d_target, target_is_left, target_2d_hand_box_center, '')
        length = (torch.max(joint_2d_target, 0)[0] - torch.min(joint_2d_target, 0)[0]).max()
        random_scale = (2.2+(np.random.rand()-0.5))  # 1.7 - 2.7
        length = int(length * random_scale)
        j = int(target_2d_hand_box_center[0].item()) - int(length/2) + int((np.random.rand()-0.5)*(length/4))
        i = int(target_2d_hand_box_center[1].item()) - int(length/2) + int((np.random.rand()-0.5)*(length/4))
        image = image.crop((j, i, j + length, i + length))

        # Scale & Translate the Target
        scale = self.transform.transforms[0].size
        joint_2d_target[:, 0] = (joint_2d_target[:, 0] - j) * (scale[0]/length)
        joint_2d_target[:, 1] = (joint_2d_target[:, 1] - i) * (scale[1]/length)
        target_2d_hand_box_center[0] = (target_2d_hand_box_center[0] - j) * (scale[0]/length)
        target_2d_hand_box_center[1] = (target_2d_hand_box_center[1] - i) * (scale[1]/length)

        # For the Right Hand Model
        if target_is_left.item():
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            joint_2d_target[:, 0] = scale[0] - joint_2d_target[:, 0]
            target_2d_hand_box_center[0] = scale[0] - target_2d_hand_box_center[0]

        # To Tensor
        if self.transform is not None:
            image = self.transform(image)

        # displayImage(index, image, joint_2d_target, target_is_left, target_2d_hand_box_center, 'cropped')
        return image, joint_2d_target, index

    def __len__(self):
        return len(self.path_image)


#
class REAL_seq3(data.Dataset):
    def __init__(self, args, mode='train', transform=None, loader=default_loader):
        self.root_real = args.root_REAL
        self.root_stereo = args.root_stereo
        self.mode = mode
        self.loader = loader
        self.transform = transform
        self.transform_temporal = TemporalTransform(3)
        # Initialize image paths
        self.path_image = make_dataset(os.path.join(self.root_real, 'images'), extensions)
        if len(self.path_image) == 0 :
            raise (RuntimeError(
                "Found 0 files in  the subfolders in path of images with supported extensions : " + ",".join(
                    extensions)))

        self.path_image = np.repeat(np.array(self.path_image).reshape(130240, 1), 3, axis=1).tolist()
        self.dataset_type = [7 for _ in range(len(self.path_image))]


        # Initialize mask paths
        # Initialize image paths - REAL
        self.path_mask = make_dataset(os.path.join(self.root_real, 'masks'), extensions)
        if len(self.path_mask) == 0 :
            raise (RuntimeError("Found 0 files in  the subfolders in path of images with supported extensions : " + ",".join(extensions)))
        self.path_mask = np.repeat(np.array(self.path_mask).reshape(130240, 1), 3, axis=1).tolist()


        # Load REAL Pickle Annotation Files
        target_real = pickle.load(open(os.path.join(self.root_real, 'ground_truths.pickle'), 'rb'))
        joint_2d_target_real = np.stack(target_real['2d'], 0).reshape(130240, 1, 21, 2)
        joint_3d_target_real = np.stack(target_real['3d'], 0).reshape(130240, 1, 21, 3)
        verts_3d_target_real = np.stack(target_real['3d_verts']).reshape(130240, 1, 778, 3)
        shape_target_real = np.stack(target_real['shapes']).reshape(130240, 1, 10)
        camera_param_target_real = np.zeros((joint_2d_target_real.shape[0], joint_2d_target_real.shape[1], 26))
        camera_param_target_real[:, :, 16:] = shape_target_real

        # To Tensor
        self.joint_2d_target = torch.from_numpy(joint_2d_target_real).repeat(1, 3, 1, 1).float()  # [n_samples,  21, 2]
        self.joint_3d_target = torch.from_numpy(joint_3d_target_real).repeat(1, 3, 1, 1).float()
        self.verts_3d_target = torch.from_numpy(verts_3d_target_real).repeat(1, 3, 1, 1).float()
        self.camera_param_target = torch.from_numpy(camera_param_target_real).repeat(1, 3, 1).float()

        if mode == 'train' :
            self.path_image = self.path_image[:-200]
            self.path_mask = self.path_mask[:-200]
            self.joint_2d_target = self.joint_2d_target[:-200]
            self.joint_3d_target = self.joint_3d_target[:-200]
            self.verts_3d_target = self.verts_3d_target[:-200]
            self.camera_param_target = self.camera_param_target[:-200]
            self.dataset_type = self.dataset_type[:-200]
        elif mode == 'valid' :
            self.path_image = self.path_image[-200:]
            self.path_mask = self.path_mask[-200:]
            self.joint_2d_target = self.joint_2d_target[-200:]
            self.joint_3d_target = self.joint_3d_target[-200:]
            self.verts_3d_target = self.verts_3d_target[-200:]
            self.camera_param_target = self.camera_param_target[-200:]
            self.dataset_type = self.dataset_type[-200:]
        # print(self.joint_3d_target.size(), self.joint_2d_target.size(), self.verts_3d_target.size(),
        #       self.camera_param_target.size(), len(self.path_image), type(self.path_image[0]))
        self.n_total_sample = len(self.path_image)

    def __getitem__(self, index):
        try :
            image = self.loader(self.path_image[index])
            mask = self.loader(self.path_mask[index])
        except :
            images = []
            masks = []
            for (img, msk) in zip(self.path_image[index], self.path_mask[index]) :
                images.append(self.loader(img))
                masks.append(self.loader(msk))
            image = images
            mask = masks
        joint_2d_target = self.joint_2d_target[index]
        joint_3d_target = self.joint_3d_target[index]
        verts_3d_target = self.verts_3d_target[index]
        camera_param_target = self.camera_param_target[index]
        dataset_type = self.dataset_type[index]

        if self.transform is not None:
            image = self.transform(image)  # [10, 3, 224, 224]
            mask = self.transform(mask)
        if self.transform_temporal is not None:
            (image, mask, joint_2d_target, joint_3d_target, verts_3d_target, camera_param_target) = self.transform_temporal((image, mask, joint_2d_target, joint_3d_target, verts_3d_target, camera_param_target))
        return image, mask, joint_2d_target, joint_3d_target, verts_3d_target, camera_param_target, dataset_type, index
    def __len__(self):
        return self.n_total_sample
#
class REAL_seq(data.Dataset):
    def __init__(self, args, mode='train', transform=None, loader=default_loader):
        self.root_real = args.root_REAL
        self.root_stereo = args.root_stereo
        self.mode = mode
        self.loader = loader
        self.transform = transform
        self.transform_temporal = TemporalTransform(args.seq_size_train)
        # Initialize image paths
        self.path_image = make_dataset(os.path.join(self.root_real, 'images'), extensions)
        if len(self.path_image) == 0 :
            raise (RuntimeError(
                "Found 0 files in  the subfolders in path of images with supported extensions : " + ",".join(
                    extensions)))

        self.path_image = np.array(self.path_image).reshape(130240, 1).tolist()
        self.dataset_type = [7 for _ in range(len(self.path_image))]


        # Initialize mask paths
        # Initialize image paths - STEREO
        # Initialize image paths - REAL
        self.path_mask = make_dataset(os.path.join(self.root_real, 'masks'), extensions)
        if len(self.path_mask) == 0 :
            raise (RuntimeError("Found 0 files in  the subfolders in path of images with supported extensions : " + ",".join(extensions)))
        self.path_mask = np.array(self.path_mask).reshape(130240, 1).tolist()


        # Load REAL Pickle Annotation Files
        target_real = pickle.load(open(os.path.join(self.root_real, 'ground_truths.pickle'), 'rb'))
        joint_2d_target_real = np.stack(target_real['2d'], 0).reshape(130240, 1, 21, 2)
        joint_3d_target_real = np.stack(target_real['3d'], 0).reshape(130240, 1, 21, 3)
        verts_3d_target_real = np.stack(target_real['3d_verts']).reshape(130240, 1, 778, 3)
        shape_target_real = np.stack(target_real['shapes']).reshape(130240, 1, 10)
        camera_param_target_real = np.zeros((joint_2d_target_real.shape[0], joint_2d_target_real.shape[1], 26))
        #camera_param_target_real[:, :, 16:] = shape_target_real

        # To Tensor
        self.joint_2d_target = torch.from_numpy(joint_2d_target_real).float()  # [n_samples,  21, 2]
        self.joint_3d_target = torch.from_numpy(joint_3d_target_real).float()  # [n_samples,  21, 3]
        self.verts_3d_target = torch.from_numpy(verts_3d_target_real).float()  # [n_samples, 778, 3]
        self.camera_param_target = torch.from_numpy(camera_param_target_real).float()  # [n_samples, 26]


        if mode == 'train' :
            self.path_image = self.path_image[:-200]
            self.path_mask = self.path_mask[:-200]
            self.joint_2d_target = self.joint_2d_target[:-200]
            self.joint_3d_target = self.joint_3d_target[:-200]
            self.verts_3d_target = self.verts_3d_target[:-200]
            self.camera_param_target = self.camera_param_target[:-200]
            self.dataset_type = self.dataset_type[:-200]
        elif mode == 'valid' :
            self.path_image = self.path_image[-200:]
            self.path_mask = self.path_mask[-200:]
            self.joint_2d_target = self.joint_2d_target[-200:]
            self.joint_3d_target = self.joint_3d_target[-200:]
            self.verts_3d_target = self.verts_3d_target[-200:]
            self.camera_param_target = self.camera_param_target[-200:]
            self.dataset_type = self.dataset_type[-200:]
        print(self.joint_3d_target.size(), self.joint_2d_target.size(), self.verts_3d_target.size(),
              self.camera_param_target.size())
        self.n_total_sample = len(self.path_image)

    def __getitem__(self, index):
        try :
            image = self.loader(self.path_image[index])
            mask = self.loader(self.path_mask[index])
        except :
            images = []
            masks = []
            for (img, msk) in zip(self.path_image[index], self.path_mask[index]) :
                images.append(self.loader(img))
                masks.append(self.loader(msk))
            image = images
            mask = masks
        joint_2d_target = self.joint_2d_target[index]
        joint_3d_target = self.joint_3d_target[index]
        verts_3d_target = self.verts_3d_target[index]
        camera_param_target = self.camera_param_target[index]
        dataset_type = self.dataset_type[index]

        if self.transform is not None:
            image = self.transform(image)  # [10, 3, 224, 224]
            mask = self.transform(mask)
        # if self.transform_temporal is not None:
            # (image, mask, joint_2d_target, joint_3d_target, verts_3d_target, camera_param_target) = self.transform_temporal((image, mask, joint_2d_target, joint_3d_target, verts_3d_target, camera_param_target))
        return image, mask, joint_2d_target, joint_3d_target, verts_3d_target, camera_param_target, dataset_type, index
    def __len__(self):
        return self.n_total_sample

class STEREO_seq(data.Dataset):
    def __init__(self, args, mode='train', transform=None, loader=default_loader, normalize = False):
        self.root = args.root_stereo
        self.mode = mode
        self.loader = loader
        self.transform = transform
        if mode == 'train':
            self.transform_temporal = TemporalTransform(args.seq_size_train)
            dir_ = 'STB_cropped_train'
        elif mode == 'valid':
            self.transform_temporal = TemporalTransform(args.seq_size_valid)
            dir_ = 'STB_cropped_valid'
        train_dir = 'STB_cropped' if mode == 'train' else 'STB_cropped_valid'
        self.path_image = make_dataset(os.path.join(self.root, dir_, 'images'), extensions)
        self.dataset_type = [1 for _ in range(len(self.path_image))]
        self.path_mask = make_dataset(os.path.join(self.root, dir_, 'masks'), extensions)
        target = pickle.load(open(os.path.join(self.root, dir_, 'gt.pickle'), 'rb'))

        if len(self.path_image) == 0:
            raise (RuntimeError("Found 0 files in subfolders of: " + os.path.join(self.root, mode, 'synth_hand_movement_data_generations') + "\n"
                                                                                                                                             "Supported extensions are: " + ",".join(extensions)))
        if len(self.path_mask) == 0:
            raise (RuntimeError("Found 0 files in subfolders of: " + os.path.join(self.root, mode, 'synth_hand_movement_data_mask') + "\n"
                                                                                                                                      "Supported extensions are: " + ",".join(extensions)))
        # Load Pickle Annotation Files
        joint_2d_target = target['2d']  # [10,  42]
        joint_3d_target = target['3d']  # [10,  21, 3]

        # Convert to Tensor
        self.joint_2d_target = torch.from_numpy(np.stack(joint_2d_target, 0)).float().view(-1, 10, 21, 2)  # [n_samples, 10,  21, 2]
        self.joint_3d_target = torch.from_numpy(np.stack(joint_3d_target, 0)).float()  # [n_samples, 10,  21, 3]
        self.camera_param = torch.zeros((self.joint_2d_target.size(0), self.joint_2d_target.size(1), 26))
        self.verts_3d_target = np.zeros((self.joint_2d_target.size(0),  self.joint_2d_target.size(1), 778, 3))
        if normalize :
            # Normalize 3D Joint Parameters
            joint_3d_target_mean, joint_3d_target_std = pickle.load(open(os.path.join(self.root, 'STEREO_train_statistics.pickle'), 'rb'))
            self.joint_3d_target = (self.joint_3d_target - joint_3d_target_mean) / joint_3d_target_std
        self.n_total_sample = len(target['2d'])
        print(self.n_total_sample)

    def __getitem__(self, index):
        image = self.loader(self.path_image[index])
        mask = self.loader(self.path_mask[index])
        joint_2d_target = self.joint_2d_target[index]
        joint_3d_target = self.joint_3d_target[index]
        camera_param_target = self.camera_param[index]
        verts_3d_target = self.verts_3d_target[index]
        dataset_type = self.dataset_type[index]
        if self.transform is not None:
            image = self.transform(image)  # [10, 3, 224, 224]
            mask = self.transform(mask)  # [10, 3, 224, 224]
        if self.transform_temporal is not None:
            (image, mask, joint_2d_target, joint_3d_target, verts_3d_target, camera_param_target) = self.transform_temporal((image, mask, joint_2d_target, joint_3d_target, verts_3d_target, camera_param_target))
        return image, mask, joint_2d_target, joint_3d_target, torch.zeros(1), camera_param_target, dataset_type, index

    def __len__(self):
        return self.n_total_sample


class HAND3D(data.Dataset):
    """
        Vanyole's Synthetic Dataset
        scale_gt = gt[0]     # dim 1   - Scale min: 684.4928, max: 1976.6282, mean: 989.7888, std: 169.2643
        trans_gt = gt[1:3]   # dim 2   - Trans min: -2.9539 / 0.1870, max: 317.1530 / 319.5916, mean: 148.9216 / 152.9533, std: 59.7917 / 61.9250
        rot_gt   = gt[3:6]   # dim 3   - Angle in [-3.14, 3.14]
        theta_gt = gt[6:16]  # dim 10  - Pose  in [-2.00, 2.00]
        beta_gt  = gt[16:]   # dim 10  - Shape in [-0.03, 0.03]
    """
    def __init__(self, root, mode='train', transform=None, loader=default_loader):
        self.size = torch.tensor([320.0, 320.0])
        self.root = root
        self.mode = mode
        self.loader = loader
        self.transform = transform
        self.path_image = make_dataset(os.path.join(self.root, mode, 'new_data'), extensions)
        if len(self.path_image) == 0:
            raise(RuntimeError("Found 0 files in subfolders of: " + os.path.join(self.root, mode, 'new_data') + "\n"
                               "Supported extensions are: " + ",".join(extensions)))
        self.path_mask = make_dataset(os.path.join(self.root, mode, 'new_data_mask'), extensions)
        if len(self.path_mask) == 0:
            raise(RuntimeError("Found 0 files in subfolders of: " + os.path.join(self.root, mode, 'new_data_mask') + "\n"
                               "Supported extensions are: " + ",".join(extensions)))
        # Load Pickle Annotation Files
        target_2d = pickle.load(open(os.path.join(self.root, mode, 'new_data_gt', '2d_joint_labels.pickle'), 'rb'))
        target_3d = pickle.load(open(os.path.join(self.root, mode, 'new_data_gt', '3d_joint_labels.pickle'), 'rb'))
        camera_param_target = pickle.load(open(os.path.join(self.root, mode, 'new_data_gt', 'camera_pose_shape_gt.pickle'), 'rb'))
        # Convert to Tensor
        self.joint_2d_target = torch.from_numpy(np.stack(target_2d, 0)).float().view(-1, 21, 2)
        self.joint_3d_target = torch.from_numpy(np.stack(target_3d['joints'], 0)).float()
        self.verts_3d_target = torch.from_numpy(np.stack(target_3d['verts'], 0)).float()
        self.camera_param_target = torch.from_numpy(np.stack(camera_param_target, 0)).float()[:, 1:]
        # Normalize Camera Parameters
        self.camera_param_target[:, 0] = (self.camera_param_target[:, 0] - self.camera_param_target[:, 0].mean()) / self.camera_param_target[:, 0].std()
        self.camera_param_target[:, 1] = (self.camera_param_target[:, 1] - self.camera_param_target[:, 1].mean()) / self.camera_param_target[:, 1].std()
        self.camera_param_target[:, 2] = (self.camera_param_target[:, 2] - self.camera_param_target[:, 2].mean()) / self.camera_param_target[:, 2].std()
        self.n_total_sample = len(target_2d)

    def __getitem__(self, index):
        image = self.loader(self.path_image[index])
        mask = self.loader(self.path_mask[index])
        joint_2d_target = self.joint_2d_target[index]
        joint_3d_target = self.joint_3d_target[index]
        verts_3d_target = self.verts_3d_target[index]
        camera_param_target = self.camera_param_target[index]
        if self.transform is not None:
            image = self.transform(image)
            mask = self.transform(mask)
            ratio = self.transform.transforms[0].size / self.size
            joint_2d_target = joint_2d_target * ratio
        return image, mask, joint_2d_target, joint_3d_target, verts_3d_target, camera_param_target, index

    def __len__(self):
        return self.n_total_sample

class HAND3D_seq(data.Dataset):
    """
        Vanyole's Sequential Synthetic Dataset
        scale_gt = gt[0]     # dim 1   - Scale min: 684.4928, max: 1976.6282, mean: 989.7888, std: 169.2643
        trans_gt = gt[1:3]   # dim 2   - Trans min: -2.9539 / 0.1870, max: 317.1530 / 319.5916, mean: 148.9216 / 152.9533, std: 59.7917 / 61.9250
        rot_gt   = gt[3:6]   # dim 3   - Angle in [-3.14, 3.14]
        theta_gt = gt[6:16]  # dim 10  - Pose  in [-2.00, 2.00]
        beta_gt  = gt[16:]   # dim 10  - Shape in [-2.00, 2.00] <<<<<--- CHANGED!!
    """
    def __init__(self, args, mode='train', transform=None, loader=default_loader, wSTB=False):
        self.size = torch.tensor([320.0, 320.0])
        self.root = args.root
        self.mode = mode
        self.loader = loader
        self.transform = transform
        if mode == 'train':
            self.transform_temporal = TemporalTransform(args.seq_size_train)
        elif mode == 'valid':
            self.transform_temporal = TemporalTransform(args.seq_size_valid)
        self.path_image = make_dataset(os.path.join(self.root, mode, 'synth_hand_movement_data_generations'), extensions)
        if len(self.path_image) == 0:
            raise(RuntimeError("Found 0 files in subfolders of: " + os.path.join(self.root, mode, 'synth_hand_movement_data_generations') + "\n"
                               "Supported extensions are: " + ",".join(extensions)))
        self.dataset_type = [0 for _ in range(len(self.path_image))]
        self.path_mask = make_dataset(os.path.join(self.root, mode, 'synth_hand_movement_data_mask'), extensions)
        if len(self.path_mask) == 0:
            raise(RuntimeError("Found 0 files in subfolders of: " + os.path.join(self.root, mode, 'synth_hand_movement_data_mask') + "\n"
                               "Supported extensions are: " + ",".join(extensions)))
        # Load Pickle Annotation Files
        if mode == 'valid':
            target = pickle.load(open(os.path.join(self.root, mode, 'synth_hand_movement_gt', 'ground_truths.pickle'), 'rb'))
            joint_2d_target = target['joints_2d']  # [10,  42]
            joint_3d_target = target['joints_3d']  # [10,  21, 3]
            verts_3d_target = target['verts_3d']  # [10, 778, 3]
            camera_param_target = target['cam_params']  # [10,  27]
            # Convert to Tensor
            self.joint_2d_target = torch.from_numpy(np.stack(joint_2d_target, 0)).float().view(-1, 10, 21, 2)   # [n_samples, 10,  21, 2]
            self.joint_3d_target = torch.from_numpy(np.stack(joint_3d_target, 0)).float()                       # [n_samples, 10,  21, 3]
            self.verts_3d_target = torch.from_numpy(np.stack(verts_3d_target, 0)).float()                       # [n_samples, 10, 778, 3]
            self.camera_param_target = torch.from_numpy(np.stack(camera_param_target, 0)).float()[:, :, 1:]     # [n_samples, 10, 26]
        else:
            target = pickle.load(open(os.path.join(self.root, mode, 'synth_hand_movement_gt', 'ground_truths.pickle'), 'rb'))
            joint_2d_target = target['joints_2d']       # [10,  42]
            joint_3d_target = target['joints_3d']       # [10,  21, 3]
            verts_3d_target = target['verts_3d']        # [10, 778, 3]
            camera_param_target = target['cam_params']  # [10,  27]
            target_10000 = pickle.load(open(os.path.join(self.root, mode, 'synth_hand_movement_gt', 'ground_truths_10000.pickle'), 'rb'))
            joint_2d_target_10000 = target_10000['joints_2d']  # [10,  42]
            joint_3d_target_10000 = target_10000['joints_3d']  # [10,  21, 3]
            verts_3d_target_10000 = target_10000['verts_3d']  # [10, 778, 3]
            camera_param_target_10000 = target_10000['cam_params']  # [10,  27]
            target_20000 = pickle.load(open(os.path.join(self.root, mode, 'synth_hand_movement_gt', 'ground_truths_20000.pickle'), 'rb'))
            joint_2d_target_20000 = target_20000['joints_2d']  # [10,  42]
            joint_3d_target_20000 = target_20000['joints_3d']  # [10,  21, 3]
            verts_3d_target_20000 = target_20000['verts_3d']  # [10, 778, 3]
            camera_param_target_20000 = target_20000['cam_params']  # [10,  27]
            joint_2d_target = np.concatenate((np.stack(joint_2d_target, 0), np.stack(joint_2d_target_10000, 0), np.stack(joint_2d_target_20000, 0)), axis=0)
            joint_3d_target = np.concatenate((np.stack(joint_3d_target, 0), np.stack(joint_3d_target_10000, 0), np.stack(joint_3d_target_20000, 0)), axis=0)
            verts_3d_target = np.concatenate((np.stack(verts_3d_target, 0), np.stack(verts_3d_target_10000, 0), np.stack(verts_3d_target_20000, 0)), axis=0)
            camera_param_target = np.concatenate((np.stack(camera_param_target, 0), np.stack(camera_param_target_10000, 0), np.stack(camera_param_target_20000, 0)), axis=0)
            # Convert to Tensor
            self.joint_2d_target = torch.from_numpy(joint_2d_target).float().view(-1, 10, 21, 2)   # [n_samples, 10,  21, 2]
            self.joint_3d_target = torch.from_numpy(joint_3d_target).float()                       # [n_samples, 10,  21, 3]
            self.verts_3d_target = torch.from_numpy(verts_3d_target).float()                       # [n_samples, 10, 778, 3]
            self.camera_param_target = torch.from_numpy(camera_param_target).float()[:, :, 1:]     # [n_samples, 10, 26]

        self.camera_s_mean = self.camera_param_target[:, :, 0].mean()
        self.camera_s_std  = self.camera_param_target[:, :, 0].std()
        self.camera_u_mean = self.camera_param_target[:, :, 1].mean()
        self.camera_u_std  = self.camera_param_target[:, :, 1].std()
        self.camera_v_mean = self.camera_param_target[:, :, 2].mean()
        self.camera_v_std  = self.camera_param_target[:, :, 2].std()
        print('Camera Parameters: Scale(mean/std) %f/%f, Translate(mean/std) %f/%f, %f/%f' % (self.camera_s_mean, self.camera_s_std, self.camera_u_mean, self.camera_u_std, self.camera_v_mean, self.camera_v_std))

        # Normalize Camera Parameters
        self.camera_param_target[:, :, 0] = (self.camera_param_target[:, :, 0] - self.camera_s_mean) / self.camera_s_std
        self.camera_param_target[:, :, 1] = (self.camera_param_target[:, :, 1] - self.camera_u_mean) / self.camera_u_std
        self.camera_param_target[:, :, 2] = (self.camera_param_target[:, :, 2] - self.camera_v_mean) / self.camera_v_std
        self.n_total_sample = self.camera_param_target.shape[0]

    def __getitem__(self, index):
        image = self.loader(self.path_image[index])
        mask = self.loader(self.path_mask[index])
        joint_2d_target = self.joint_2d_target[index]
        joint_3d_target = self.joint_3d_target[index]
        verts_3d_target = self.verts_3d_target[index]
        camera_param_target = self.camera_param_target[index]
        dataset_type = self.dataset_type[index]
        if self.transform is not None:
            image = self.transform(image)  # [10, 3, 224, 224]
            mask = self.transform(mask)  # [10, 3, 224, 224]
        if self.transform_temporal is not None:
            (image, mask, joint_2d_target, joint_3d_target, verts_3d_target, camera_param_target) = self.transform_temporal((image, mask, joint_2d_target, joint_3d_target, verts_3d_target, camera_param_target))
        return image, mask, joint_2d_target, joint_3d_target, verts_3d_target, camera_param_target, dataset_type, index

    def __len__(self):
        return self.n_total_sample



class RWORLD_seq(data.Dataset):
    def __init__(self, args, mode='train', transform=None, loader=pil_loader_rworld, normalize = False):
        self.size = torch.tensor([320.0, 320.0])
        self.root_real = args.root_real
        self.mode = mode
        self.loader = loader
        self.transform = transform
        self.transform_temporal = TemporalTransform(args.seq_size_train)

        # Initialize image paths
        self.path_image = make_dataset(os.path.join(self.root_real, 'images'), extensions)
        self.dataset_type = [2 for _ in range(len(self.path_image))]

        # Initialize mask paths
        # Initialize image paths - STEREO
        self.path_mask = make_dataset(os.path.join(self.root_real, 'masks'), extensions)
        if len(self.path_image) == 0 :
            raise (RuntimeError("Found 0 files in  the subfolders in path of images with supported extensions : " + ",".join(extensions)))

        # Load REAL Pickle Annotation Files
        target_real = pickle.load(open(os.path.join(self.root_real, 'gt.pickle'), 'rb'))
        joint_2d_target_real = np.stack(target_real['2d'], 0).reshape(238, 10, 42)
        joint_3d_target_real = np.stack(target_real['3d'], 0)
        verts_3d_target_real = np.zeros((joint_2d_target_real.shape[0], joint_2d_target_real.shape[1], 778, 3))
        camera_param_target_real = np.zeros((joint_2d_target_real.shape[0], joint_2d_target_real.shape[1], 27))
        # Convert to Tensor
        self.joint_2d_target = torch.from_numpy(joint_2d_target_real).float().view(-1, 10, 21, 2)  # [n_samples, 10,  21, 2]
        self.joint_3d_target = torch.from_numpy(joint_3d_target_real).float()  # [n_samples, 10,  21, 3]

        #
        # self.joint_3d_target_min0 = self.joint_3d_target[:, :, :, 0].min()
        # self.joint_3d_target_min1 = self.joint_3d_target[:, :, :, 1].min()
        # self.joint_3d_target_min2 = self.joint_3d_target[:, :, :, 2].min()
        # self.joint_3d_target_rnge0 = self.joint_3d_target[:, :, :, 0].max() - self.joint_3d_target_min0
        # self.joint_3d_target_rnge1 = self.joint_3d_target[:, :, :, 1].max() - self.joint_3d_target_min1
        # self.joint_3d_target_rnge2 = self.joint_3d_target[:, :, :, 2].max() - self.joint_3d_target_min2
        #

        self.verts_3d_target = torch.from_numpy(verts_3d_target_real).float()  # [n_samples, 10, 778, 3]
        self.camera_param_target = torch.from_numpy(camera_param_target_real).float()[:, :, 1:]  # [n_samples, 10, 26]
        self.dataset_type = torch.from_numpy(np.stack(self.dataset_type, 0))  # 1: STEREO, 2: REAL, 3: HAND3D

        self.n_total_sample = self.camera_param_target.shape[0]


    def __getitem__(self, index):
        image = self.loader(self.path_image[index])
        # import time
        # image[0].show()
        # time.sleep(15)

        mask = self.loader(self.path_mask[index])
        joint_2d_target = self.joint_2d_target[index]
        joint_3d_target = self.joint_3d_target[index]
        verts_3d_target = self.verts_3d_target[index]
        camera_param_target = self.camera_param_target[index]
        dataset_type = self.dataset_type[index]
        if self.transform is not None:
            image = self.transform(image)  # [10, 3, 224, 224]
            mask = self.transform(mask)  # [10, 3, 224, 224]
        if self.transform_temporal is not None:
            (image, mask, joint_2d_target, joint_3d_target, verts_3d_target, camera_param_target) = self.transform_temporal((image, mask, joint_2d_target, joint_3d_target, verts_3d_target, camera_param_target))
        return image, mask, joint_2d_target, joint_3d_target, verts_3d_target, camera_param_target, dataset_type, index

    def __len__(self):
        return self.n_total_sample


class EgoDexter_seq(data.Dataset):
    def __init__(self, args, mode='train', transform=None, loader=default_loader, normalize = False):
        self.root = args.root_egodexter
        self.mode = mode
        self.loader = loader
        self.transform = transform
        self.transform_temporal = TemporalTransform(-1)
        self.path_image = make_dataset(os.path.join(self.root, 'images'), extensions)
        self.dataset_type = [4 for _ in range(len(self.path_image))]  # 4: EgoDexter
        target = pickle.load(open(os.path.join(self.root, 'gt_ED.pickle'), 'rb'))
        if len(self.path_image) == 0:
            raise (RuntimeError("Found 0 files in subfolders of: " + os.path.join(self.root, 'images') + "\n" "Supported extensions are: " + ",".join(extensions)))
        # Load Pickle Annotation Files
        self.joint_2ds_target = target['2ds']  # 353 * [n_seq, 5, 2], n_seq in {6, 7, 8, 10, ... }
        self.joint_3ds_target = target['3ds']  # 353 * [n_seq, 5, 3]
        self.joint_2d_clean_target = target['2d_clean']  # 353 * [n_seq, 5, 2], n_seq in {6, 7, 8, 10, ... }
        self.joint_3d_clean_target = target['3d_clean']  # 353 * [n_seq, 5, 3]

        # Convert to Tensor
        for idx in range(len(self.joint_2ds_target)):
            self.joint_2ds_target[idx] = torch.from_numpy(self.joint_2ds_target[idx]).float()
            self.joint_3ds_target[idx] = torch.from_numpy(self.joint_3ds_target[idx]).float()
            self.joint_2d_clean_target[idx] = torch.from_numpy(self.joint_2d_clean_target[idx]).float()
            self.joint_3d_clean_target[idx] = torch.from_numpy(self.joint_3d_clean_target[idx]).float()
        self.camera_param = torch.zeros((len(self.joint_2ds_target), 10, 26))
        # Normalize 3D Joint Parameters
        joint_3ds_target_mean = torch.cat(self.joint_3ds_target).mean(0)
        joint_3ds_target_std = torch.cat(self.joint_3ds_target).std(0)
        joint_3d_clean_target_mean = torch.cat(self.joint_3d_clean_target).mean(0)
        joint_3d_clean_target_std = torch.cat(self.joint_3d_clean_target).std(0)
        with open(os.path.join(self.root, 'EGODEXTER_train_statistics.pickle'), 'wb') as f:
            pickle.dump([joint_3ds_target_mean, joint_3ds_target_std, joint_3d_clean_target_mean, joint_3d_clean_target_std], f)
        if normalize :
            for idx in range(len(self.joint_2ds_target)):
                self.joint_3d_clean_target[idx] = (self.joint_3d_clean_target[idx] - joint_3d_clean_target_mean) / joint_3d_clean_target_std
                self.joint_3ds_target[idx] = (self.joint_3ds_target[idx] - joint_3ds_target_mean) / joint_3ds_target_std
        self.n_total_sample = len(target['2ds'])

    def __getitem__(self, index):
        image = self.loader(self.path_image[index])
        joint_2d_target = self.joint_2d_clean_target[index]
        joint_3d_target = self.joint_3d_clean_target[index]
        camera_param_target = self.camera_param[index]
        dataset_type = self.dataset_type[index]

        if self.transform is not None:
            image = self.transform(image)  # [n_seq, 3, 224, 224]
        if self.transform_temporal is not None:
            image = self.transform_temporal(image)
            joint_2d_target = self.transform_temporal(joint_2d_target)
            joint_3d_target = self.transform_temporal(joint_3d_target)
        return image, torch.zeros_like(image), joint_2d_target, joint_3d_target, torch.zeros(1), camera_param_target, dataset_type, index

    def __len__(self):
        return self.n_total_sample


class DexterObject_seq(data.Dataset):
    def __init__(self, args, mode='train', transform=None, loader=default_loader, normalize=False):
        self.root = args.root_dexterobject
        self.mode = mode
        self.loader = loader
        self.transform = transform
        self.transform_temporal = TemporalTransform(-1)
        self.path_image = make_dataset(os.path.join(self.root, 'images'), extensions)
        self.dataset_type = [5 for _ in range(len(self.path_image))]  # 5: DexterObject
        target = pickle.load(open(os.path.join(self.root, 'gt_DO.pickle'), 'rb'))
        if len(self.path_image) == 0:
            raise (RuntimeError("Found 0 files in subfolders of: " + os.path.join(self.root, 'images') + "\n" "Supported extensions are: " + ",".join(extensions)))
        # Load Pickle Annotation Files
        self.joint_2ds_target = target['2ds']  # 361 * [n_seq, 5, 2], n_seq in {6, 7, 8, 10, ... }
        self.joint_3ds_target = target['3ds']  # 361 * [n_seq, 5, 3]

        # Convert to Tensor
        for idx in range(len(self.joint_2ds_target)):
            self.joint_2ds_target[idx] = torch.from_numpy(self.joint_2ds_target[idx]).float()
            self.joint_3ds_target[idx] = torch.from_numpy(self.joint_3ds_target[idx]).float()
        self.camera_param = torch.zeros((len(self.joint_2ds_target), 10, 26))
        # Normalize 3D Joint Parameters
        joint_3ds_target_mean = torch.cat(self.joint_3ds_target).mean(0)
        joint_3ds_target_std = torch.cat(self.joint_3ds_target).std(0)
        with open(os.path.join(self.root, 'DEXTER_train_statistics.pickle'), 'wb') as f:
            pickle.dump([joint_3ds_target_mean, joint_3ds_target_std], f)
        if normalize :
            for idx in range(len(self.joint_2ds_target)):
                self.joint_3ds_target[idx] = (self.joint_3ds_target[idx] - joint_3ds_target_mean) / joint_3ds_target_std
        self.n_total_sample = len(target['2ds'])

    def __getitem__(self, index):
        image = self.loader(self.path_image[index])
        joint_2d_target = self.joint_2ds_target[index]
        joint_3d_target = self.joint_3ds_target[index]
        camera_param_target = self.camera_param[index]
        dataset_type = self.dataset_type[index]
        if self.transform is not None:
            image = self.transform(image)  # [n_seq, 3, 224, 224]
        if self.transform_temporal is not None:
            image = self.transform_temporal(image)
            joint_2d_target = self.transform_temporal(joint_2d_target)
            joint_3d_target = self.transform_temporal(joint_3d_target)
        return image, torch.zeros_like(image), joint_2d_target, joint_3d_target, torch.zeros(1), camera_param_target, dataset_type, index

    def __len__(self):
        return self.n_total_sample


class TemporalTransform(object):
    def __init__(self, seq_size):
        self.seq_size = seq_size

    def __call__(self, x):
        if self.seq_size < 0:
            return x
        else:
            n_seq = x[0].size(0)
            if n_seq == self.seq_size:
                return x
            else:
                idx_start = int(np.random.randint(low=0, high=n_seq-self.seq_size+1, size=1))
                idx_end = int(idx_start + self.seq_size)
                return [out[idx_start:idx_end] for out in x]


def displayPANOPTIC():
    import matplotlib
    import matplotlib.pyplot as plt
    import torchvision.transforms as transforms
    transform_train = transforms.Compose([transforms.ToTensor(),
                                          transforms.Normalize([0, 0, 0], [1, 1, 1])])
    # Initialize Dataset
    root = '/home/user/vanyole/HPE_0904/hpe_data/PANOPTIC'
    dataset_train = PANOPTIC(root, mode='train', transform=transform_train)
    train_loader = torch.utils.data.DataLoader(dataset=dataset_train, batch_size=16, shuffle=True,  sampler=None, num_workers=8, pin_memory=True, drop_last=True)
    index = 9
    edges = [[0, 1], [1, 2], [2, 3], [3, 4], [0, 5], [5, 6], [6, 7], [7, 8], [0, 9], [9, 10], [10, 11], [11, 12], [0, 13], [13, 14], [14, 15], [15, 16], [0, 17], [17, 18], [18, 19], [19, 20]]
    imgPath = os.path.join(dataset_train.root, dataset_train.target['img_paths'][index])
    im = plt.imread(imgPath)
    plt.imshow(im)
    for p in range(dataset_train.joint_2d_target[index].shape[0]):
        if dataset_train.joint_2d_target[index][p, 2] != 0:
            plt.plot(dataset_train.joint_2d_target[index][p, 0], dataset_train.joint_2d_target[index][p, 1], 'r.')
            plt.text(dataset_train.joint_2d_target[index][p, 0], dataset_train.joint_2d_target[index][p, 1], '{0}'.format(p))
    for ie, e in enumerate(edges):
        if np.all(dataset_train.joint_2d_target[index][e, 2] != 0):
            rgb = matplotlib.colors.hsv_to_rgb([ie / float(len(edges)), 1.0, 1.0])
            plt.plot(dataset_train.joint_2d_target[index][e, 0], dataset_train.joint_2d_target[index][e, 1], color=rgb)
    plt.axis('off')
    plt.show()

import imageio

def inside_polygon(x, y, points):
    n = len(points)
    inside = False
    p1x, p1y = points[0]
    for i in range(1, n + 1):
        p2x, p2y = points[i % n]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= xinters:
                        inside = not inside
        p1x, p1y = p2x, p2y
    return inside


if __name__ == "__main__":
    import matplotlib
    import matplotlib.pyplot as plt
    import torchvision.transforms as transforms

    root = '/media/vanyole/VANYOLE/HPE/Dataset/FHAD'
    dataset = FHAD(root, mode='train')
    train_loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=1, shuffle=False,  sampler=None,
                                               num_workers=12, pin_memory=True, drop_last=False)
    direc = dataset.new_img_direc
    for i , data in enumerate(train_loader):
        images, _, _, index = data
        new_ = []
        for image in images:
            new_.append(image.squeeze(0).numpy())

        imageio.mimsave(direc + '/%08d.gif' % index, new_)

        print(index, ' / ', len(dataset.path_img_seq))

    dataset = FHAD(root, mode='valid')
    valid_loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=1, shuffle=False, sampler=None,
                                               num_workers=12, pin_memory=True, drop_last=False)
    direc = dataset.new_img_direc
    for i, data in enumerate(valid_loader):
        images, _, _, index = data
        new_ = []
        for image in images:
            new_.append(image.squeeze(0).numpy())

        imageio.mimsave(direc + '/%08d.gif' % index, new_)

        print(index, ' / ',len(dataset.path_img_seq))
    # for seq_ind, image_paths in enumerate(self.path_img_seq):
    #     print(seq_ind, ' / ', len(self.path_img_seq))
    #     print(image_paths)
    #     images = []
    #     for img_path_ind, image_path in enumerate(image_paths):
    #         image = self.loader(image_path)
    #         joint_2d_crop, (x, y, w, l) = self.crop_around_hand(self.targ_2d_seq[seq_ind][img_path_ind])
    #         image = np.array(image.crop((y, x, y + l, x + w)).resize((224, 224)))
    #         images.append(image)
    #     imageio.mimsave(self.new_img_direc + '/%08d.gif' % seq_ind, images)
    #     # images[0].save(self.new_img_direc + '/%08d.tif' % seq_ind, save_all=True, append_images=images[1:])
    #
    # targets = {'3ds' : self.targ_3d_seq, '2ds' : self.targ_2d_seq}
    # with open(os.path.join(self.root, 'cross_sub', self.mode, 'gt.pickle'), 'wb') as f:
    #     pickle.dump(targets, f)