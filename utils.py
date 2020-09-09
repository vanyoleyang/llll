import math
import torch
import shutil
import time
import os
import random
import pickle
from easydict import EasyDict as edict
import yaml
import numpy as np
import matplotlib.pyplot as plt
from manopth.manolayer import ManoLayer
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
#from opendr.renderer import ColoredRenderer
#from opendr.lighting import LambertianPointLight
#from opendr.camera import ProjectPoints
from PIL import Image

from manopth.manolayer import ManoLayer
from manopth import demo


STB_results = {}
STB_results['Iqbal et al.'] = np.array([[20., 0.9625], [23.33333333, 0.9818], [26.66666667, 0.9908], [30., 0.9949], [33.33333333, 0.9969],[36.66666667, 0.9982], [40., 0.9988], [43.33333333, 0.9993], [46.66666667, 0.9996], [50., 0.9998]])
STB_results['Boukhayma et al.(RGB/best)'] = np.array([range(0, 51), [0.047619047619,0.048873015873,0.0570158730159, 0.077380952381,0.109365079365,0.157253968254,0.218396825397,0.291349206349,0.377396825397,0.474682539683,0.563079365079,0.643857142857,0.715523809524,0.776619047619,0.827222222222,0.86673015873,0.897285714286,0.921428571429,0.939571428571,0.953206349206,0.962857142857,0.970285714286,0.975492063492,0.979936507937,0.98326984127,0.986492063492,0.989301587302,0.99126984127,0.993142857143,0.994222222222,0.995444444444,0.996142857143,0.996825396825,0.997380952381,0.997857142857,0.998301587302,0.998571428571,0.998793650794,0.998984126984,0.999222222222,0.999349206349,0.999476190476,0.999571428571,0.999666666667,0.999682539683,0.999746031746,0.999777777778,0.999825396825,0.999841269841,0.999857142857,0.999873015873]]).T
STB_results['Cai et al.'] = np.array([[20., 0.9625], [23.33333333, 0.9818], [26.66666667, 0.9908], [30., 0.9949], [33.33333333, 0.9969],[36.66666667, 0.9982], [40., 0.9988 ], [43.33333333, 0.9993], [46.66666667, 0.9996], [50., 0.9998]])
STB_results['Mueller et al.'] = np.array([[19.1919, 0.8713], [22.2222, 0.9035], [25.2525, 0.9271],[28.2828, 0.9446], [31.3131, 0.9574], [34.3434, 0.9670],[37.3737, 0.9741], [40.4040, 0.9795], [43.4343, 0.9833],[46.4646, 0.9867], [49.4949, 0.9895]])
STB_results['Spurr et al.'] = np.array([[21.05263158, 0.94828571], [23.68421053, 0.96146825], [26.31578947, 0.97045238], [28.94736842, 0.97711905],[31.57894737, 0.98188889], [34.21052632, 0.98569841], [36.84210526, 0.98859524], [39.47368421, 0.99088095],[42.10526316, 0.99263492], [44.73684211, 0.99423016], [47.36842105, 0.99536508], [50.00000000, 0.99630159]])
STB_results['Zimmermann and Brox'] = np.array([[21.0526315789474, 0.869888888888889], [23.6842105263158, 0.896873015873016], [26.3157894736842, 0.916849206349206], [28.9473684210526, 0.932142857142857], [31.5789473684211, 0.943507936507937], [34.2105263157895, 0.952753968253968], [36.8421052631579, 0.959904761904762], [39.4736842105263, 0.966047619047619], [42.1052631578947, 0.971595238095238], [44.7368421052632, 0.976547619047619], [47.3684210526316, 0.980174603174603], [50., 0.983277777777778]])
STB_results['Panteleris et al.'] = np.array([[22, 0.612], [24, 0.796], [26, 0.8706666666666667], [28, 0.892], [30, 0.9226666666666666], [32, 0.9493333333333334], [34, 0.9593333333333334], [36, 0.9793333333333333],[38, 0.9953333333333333], [40, 1.0], [42, 1.0], [44, 1.0], [46, 1.0], [48, 1.0], [50, 1.0]])
STB_results['ICPPSO'] = np.array([[20, 0.519736842105263], [25, 0.644736842105263], [30, 0.717105263157895], [35, 0.773026315789474], [40, 0.809210526315789], [45, 0.848684210526316], [50, 0.868421052631579]])
STB_results['PSO'] = np.array([[20, 0.322368421052632], [25, 0.539473684210526], [30, 0.674342105263158], [35, 0.756578947368421], [40, 0.809210526315789], [45, 0.865131578947368], [50, 0.894736842105263]])


ED_results = {}
ED_results['Zimmermann and Brox'] = np.array([range(0,51), [0.0113517289556, 0.0113517289556,0.0117010129235,0.0134474327628,0.0146699266504,0.0197345441844,0.023402025847,0.026021655606,0.0326580509955,0.0434858539993,0.0544882989871,0.0653161019909,0.080684596577,0.0983234369542,0.113517289556,0.13098148795,0.150017464198,0.166608452672,0.187041564792,0.207125392944,0.227907789032,0.250960530912,0.276458260566,0.300034928397,0.325881942019,0.352078239609,0.377226685295,0.399930143206,0.425602514845,0.456514146001,0.483932937478,0.506287111422,0.529514495285,0.552217953196,0.565665385959,0.580684596577,0.595529165211,0.610199091862,0.624170450576, 0.639538945162,0.658050995459,0.671673070206,0.684072651065,0.696646873908,0.710618232623,0.720747467691,0.731225986727,0.740831295844,0.749912679008,0.758644778205, 0.764582605658]]).T
ED_results['Boukhayma et al.(best)'] = np.array([range(0, 51), [0.0113517289556,0.0115263709396,0.0120502968914,0.0141460006986,0.0176388403772,0.0246245197345,0.0328326929794,0.0424380020957,0.0544882989871,0.0716032134125,0.0901152637094,0.110897659797,0.137443241355,0.165735242752,0.192455466294,0.219524973804,0.24956339504,0.282046804052,0.31662591687,0.350331819769,0.383339154733,0.411281872162,0.439748515543,0.466992665037,0.491617184771,0.51466992665,0.53772266853,0.563744324136,0.586098498079,0.607230178135,0.626964722319,0.641809290954,0.66119455117,0.677261613692,0.695075096053,0.706426825009,0.721446035627,0.733670974502,0.747816975201,0.758994062173,0.767027593433,0.775585050646,0.782046804052,0.790604261264,0.799685644429,0.805972755851,0.813307719176,0.82238910234,0.829898707649,0.836011177087,0.840027942717]]).T
ED_results['Boukhayma et al.(RGB)'] = np.array([range(0, 51), [0.0113517289556,0.0115263709396,0.0118756549074,0.0123995808592,0.0137967167307,0.0158924205379,0.0200838281523,0.0254977296542,0.0319594830597,0.0426126440796,0.0527418791477,0.0689835836535,0.0887181278379,0.108278030038,0.128361858191,0.152986377925,0.178134823612,0.202235417394,0.231575270695,0.255501222494,0.277506112469,0.300558854349,0.323087670276,0.349807893818,0.37565490744,0.400803353126,0.425078588893,0.449179182676,0.471009430667,0.491617184771,0.512050296891,0.530387705204,0.549074397485,0.570904645477,0.590115263709,0.604261264408,0.617359413203,0.631156129934,0.64652462452,0.658749563395,0.671323786238,0.682675515194,0.696472231925,0.707649318896,0.720747467691,0.729304924904,0.736639888229,0.745721271394,0.753754802655,0.762312259867,0.768075445337]]).T
ED_results['Iqbal et al.'] = np.array([[0. , 0.0000], [5.26315789 , 0.0566],[10.52631579, 0.1498],[15.78947368,0.2501],[21.05263158, 0.3380],[26.31578947,  0.4203],[31.57894737,0.4942],[36.84210526, 0.5608],[42.10526316, 0.6105],[47.36842105, 0.6576],[52.63157895,     0.6955],[57.89473684,     0.7238],[63.15789474,     0.7448],[68.42105263,     0.7655],[73.68421053,     0.7893],[78.94736842,     0.8098],[84.21052632,     0.8290],[89.47368421,     0.8479],[94.73684211,     0.8646],[100.       ,     0.8804]])
ED_results['Spurr et al.'] = np.array([range(0, 51), [0.0, 0.000176647235471,0.000176647235471,0.000353294470942,0.00141317788377,0.00388623918036,0.00618265324148,0.00989224518636,0.015368309486,0.0231407878467,0.0321497968557,0.0413354531002,0.0506977565801,0.0595301183536,0.0740151916623,0.0865571453807,0.0985691573927,0.114997350291,0.130189012542,0.144497438615,0.164281928988,0.184066419361,0.202614379085,0.226108461403,0.251192368839,0.269033739622,0.294824236001,0.316375198728,0.33580639463,0.358063946299,0.377495142201,0.398869457693,0.418830595301,0.442148030383,0.464935523759,0.486839780957,0.507154213037,0.524112347642,0.54018724607,0.553965730436,0.569510687158,0.587175410705,0.604663487016,0.617558735206,0.629924041689,0.643702526055,0.655184596361,0.667019961138,0.677088853559,0.68839427663,0.704469175057]]).T

DO_results={}
DO_results['Mueler et al.'] = np.array([[ 5.0505, 0.0339],[10.1010, 0.1562],[15.1515, 0.2887],[20.2020, 0.4092],[25.2525, 0.5084],[30.3030, 0.6029],[35.3535, 0.6852],[40.4040, 0.7566],[45.4545, 0.8080],[50.5051, 0.8463],[55.5556, 0.8769],[60.6061, 0.9016],[65.6566, 0.9187],[70.7071, 0.9300],[75.7576, 0.9396],[80.8081, 0.9467],[85.8586, 0.9536],[90.9091, 0.9625],[95.9596, 0.9698]])
DO_results['Boukhayma et al.(RGB)'] = np.array([range(0, 51), [0.0,0.000141282848262,0.000494489968918,0.00289629838938,0.00692285956485,0.0122916077988,0.019920881605,0.0287510596214,0.0399830460582,0.0532636337949,0.0676038428935,0.086747668833,0.105326363379,0.123905057926,0.145591975134,0.170740322125,0.195676744843,0.224781011585,0.253532071207,0.281859282283,0.310115851936,0.337454083074,0.363167561458,0.389092964114,0.417420175191,0.441579542244,0.466021474993,0.490463407742,0.512786097768,0.533625317886,0.555382876519,0.577564283696,0.599604408025,0.620302345295,0.642130545352,0.66141565414,0.682184232834,0.700480361684,0.716515964962,0.732269002543,0.747244984459,0.761090703589,0.772958462843,0.784614297824,0.796199491382,0.807855326363,0.818168974287,0.827988132241,0.835970613168,0.843176038429,0.850734670811]]).T
DO_results['Boukhayma et al.(best)'] = np.array([range(0, 51), [0.0,0.000353207120656,0.00459169256852,0.0140576434021,0.0277620796835,0.045775642837,0.0665442215315,0.0913393614015,0.11542808703,0.14142413111,0.167490816615,0.198926250353,0.234035038146,0.268578694546,0.301073749647,0.330813789206,0.3618253744,0.392836959593,0.424625600452,0.452882170105,0.482975416784,0.510172365075,0.538923424696,0.564071771687,0.587454083074,0.611048318734,0.633017801639,0.656541395875,0.677733823114,0.695182254874,0.714538005086,0.730926815485,0.745973438825,0.761090703589,0.776278609777,0.788640859,0.801356315343,0.813294716021,0.825021192427,0.835688047471,0.845365922577,0.854337383442,0.862955637186,0.869807855326,0.877719694829,0.884359988697,0.89107092399,0.89552133371,0.901455213337,0.906046905906,0.911627578412]]).T
DO_results['Zimmermann and Brox'] = np.array([range(0, 51), [0.0,0.0,0.0,0.0,0.000494489968918,0.0011302627861,0.00240180842046,0.00416784402374,0.00741734953377,0.0117264764058,0.0187199773947,0.0268437411698,0.0384289347273,0.0540406894603,0.0702175755863,0.0917632099463,0.115216162758,0.144179146652,0.173495337666,0.201893190167,0.233681831026,0.261938400678,0.291113308844,0.317957050014,0.342469624188,0.366417066968,0.388103984176,0.410426674202,0.431901667138,0.454436281435,0.478242441368,0.501483469907,0.524230008477,0.547894885561,0.571206555524,0.594165018367,0.614933597061,0.634148064425,0.653857021758,0.671446736366,0.68946029952,0.705778468494,0.721390223227,0.737213902232,0.752119242724,0.765187906188,0.78009324668,0.794221531506,0.808561740605,0.822831308279,0.835829330319]]).T
DO_results['Iqbal et al.'] = np.array([[0.0,0.0000],[5.26315789 ,0.0619],[10.52631579,0.1872],[15.78947368,0.3071],[21.05263158,0.4142],[26.31578947,0.5102],[31.57894737,0.6024],[36.84210526,0.6843],[42.10526316,0.7634],[47.36842105,0.8268],[52.63157895,0.8784],[57.89473684,0.9179],[63.15789474,0.9466],[68.42105263,0.9652],[73.68421053,0.9775],[78.94736842,0.9845],[84.21052632,0.9881],[89.47368421,0.9925],[94.73684211,0.9950],[100.0,0.9964]])
DO_results['Spurr et al.'] = np.array([range(0, 51), [0.0,0.000282565696524,0.00155411133088,0.00480361684092,0.00932466798531,0.0146934162193,0.0221814071772,0.0305877366488,0.0397711217858,0.0520627295846,0.0633653574456,0.0773523594236,0.0912687199774,0.104831873411,0.120867476688,0.136691155694,0.152585476123,0.168691720825,0.186281435434,0.203164735801,0.223862673071,0.243147781859,0.263775077706,0.286521616276,0.306301215032,0.326787228031,0.348615428087,0.367829895451,0.388315908449,0.408236790054,0.427733823114,0.449632664595,0.467363662051,0.486436846567,0.504874258265,0.523382311387,0.54104266742,0.55813789206,0.575656965244,0.593246679853,0.609635490251,0.626801356315,0.642695676745,0.657247810116,0.669610059339,0.683597061317,0.69595931054,0.707120655552,0.715738909296,0.728030517095,0.737990957898]]).T


class AverageMeter(object):
    """ Computes ans stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0.
        self.avg = 0.
        self.sum = 0.
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(optimizer, iters, base_lr, policy_parameter=None, policy='step', multiple=[1]):
    if policy == 'fixed':
        lr = base_lr
    elif policy == 'step':
        lr = base_lr * (policy_parameter['gamma'] ** (iters // policy_parameter['step_size']))
    elif policy == 'exp':
        lr = base_lr * (policy_parameter['gamma'] ** iters)
    elif policy == 'inv':
        lr = base_lr * ((1 + policy_parameter['gamma'] * iters) ** (-policy_parameter['power']))
    elif policy == 'multistep':
        lr = base_lr
        for stepvalue in policy_parameter['stepvalue']:
            if iters >= stepvalue:
                lr *= policy_parameter['gamma']
            else:
                break
    elif policy == 'poly':
        lr = base_lr * ((1 - iters * 1.0 / policy_parameter['max_iter']) ** policy_parameter['power'])
    elif policy == 'sigmoid':
        lr = base_lr * (1.0 / (1 + math.exp(-policy_parameter['gamma'] * (iters - policy_parameter['stepsize']))))
    elif policy == 'multistep-poly':
        lr = base_lr
        stepstart = 0
        stepend = policy_parameter['max_iter']
        for stepvalue in policy_parameter['stepvalue']:
            if iters >= stepvalue:
                lr *= policy_parameter['gamma']
                stepstart = stepvalue
            else:
                stepend = stepvalue
                break
        lr = max(lr * policy_parameter['gamma'], lr * (1 - (iters - stepstart) * 1.0 / (stepend - stepstart)) ** policy_parameter['power'])

    for i, param_group in enumerate(optimizer.param_groups):
        param_group['lr'] = lr * multiple[i]
    return lr


def save_checkpoint(state, is_best, filename='checkpoint'):
    torch.save(state, filename + '_latest.pth.tar')
    if is_best:
        shutil.copyfile(filename + '_latest.pth.tar', filename + '_best.pth.tar')


def load_checkpoint(folder_name, filename='checkpoint'):
    model = torch.load(os.path.join(folder_name, filename))
    return model


def Config(filename):
    with open(filename, 'r') as f:
        parser = edict(yaml.safe_load(f))
    for x in parser:
        print('{}: {}'.format(x, parser[x]))
    return parser


def wrap_surfaces(hand_info, mano_faces=None, ax=None, alpha=1., batch_idx=0):
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
    else:
        fig = ax
        ax = fig.add_subplot(111, projection='3d')

    verts, joints = hand_info['verts'][batch_idx], hand_info['joints'][
        batch_idx]

    if mano_faces is None:
        ax.scatter(verts[:, 0], verts[:, 1], verts[:, 2], alpha=0.1)
    else:
        mesh = Poly3DCollection(verts[mano_faces],linewidths=0.5,  alpha=alpha)  # alpha = opacity
        face_color = (0.95, 0.95, 0.95)
        edge_color = (0.2, 0.2, 0.2)
        # edge_color = (50 / 255, 50 / 255, 50 / 255)
        mesh.set_edgecolor(edge_color)
        mesh.set_facecolor(face_color)
        ax.add_collection3d(mesh)
    # ax.scatter(joints[:, 0], joints[:, 1], joints[:, 2], color='r')
    demo.cam_equal_aspect_3d(ax, verts.numpy())
    axes = plt.gca()
    axes.set_xlim([0, 300])
    axes.set_ylim([-100, 100])
    axes.set_zlim([-100, 100])
    return ax


def displayImage(args, epoch, i, image, predictions, targets, mode):
    # Initialize directory
    directory = os.path.join('results', args.model_name + '_' + args.model_id, 'images')
    makeDir(directory)
    makeDir(os.path.join('results', args.model_name + '_' + args.model_id, 'images', '2D_skel'))
    makeDir(os.path.join('results', args.model_name + '_' + args.model_id, 'images', 'mesh1'))
    makeDir(os.path.join('results', args.model_name + '_' + args.model_id, 'images', 'mesh2'))

    # Initialize predictions & targets
    if args.model_name == 'EncoderConvLSTM_stoch':
        x2d_pred, x3d_pred, camera_param_pred, theta, beta, _, _ = predictions
    else : x2d_pred, x3d_pred, camera_param_pred, theta, beta = predictions
    joint_2d_target, _, _, _, _ = targets
    joint_2d_pred = torch.stack((x2d_pred[:, :, :42:2], x2d_pred[:, :, 1:42:2]), dim=3)  # x_hat

    mano_layer = ManoLayer(mano_root='/home/vanyole/Research/HPE/3dhand_sequence_20191022_findthebestframework/3dhand_sequence_20191022_untouched/mano/models',
                           use_pca=True, ncomps=10, flat_hand_mean=False, side='right').cuda()
    scale = camera_param_pred[:, :, 0]  # Scale       1
    trans = camera_param_pred[:, :, 1:3]  # Translate   2
    rotation = camera_param_pred[:, :, 3:6]  # Rotation    3
    theta_ = theta
    theta_ = theta_.view(theta.size(0) * theta.size(1), 10)
    theta_ = torch.cat((torch.tensor([math.pi, 0., 0.]).unsqueeze(0).repeat(theta.size(0) * theta.size(1), 1).float().cuda(), theta_), 1)  # [pi, 0, 0, theta]
    shape_param = torch.rand(1, 10).expand(theta.size(0) * theta.size(1), -1).cuda().float() * 3. - 1.5
    beta = beta.view(theta.size(0) * theta.size(1), 10)
    hand_verts, hand_joints = mano_layer(theta_.clone(), shape_param.clone())
    # hand_joints = torch.stack((x3d_pred[:, :, :63:3], x3d_pred[:, :, 1:63:3], x3d_pred[:, :, 2:63:3]), dim=3)
    hand_joints = hand_joints.view(theta.size(0), theta.size(1), 21, 3)
    # hand_verts = torch.stack((x3d_pred[:, :, 63::3], x3d_pred[:, :, 64::3], x3d_pred[:, :, 65::3]), dim=3)
    hand_verts = hand_verts.view(theta.size(0), theta.size(1), 778, 3)
    verts_2d = x2d_pred[:, :, 42:].view(theta.size(0), theta.size(1), 778, 2)

    # Save images
    batch_size = min(image.size(0), 4)
    seq_size = image.size(1)
    n_joints = joint_2d_target.size(2)

    color = hand_colors()[0]
    u_p = np.zeros(n_joints)
    v_p = np.zeros(n_joints)
    u_t = np.zeros(n_joints)
    v_t = np.zeros(n_joints)
    for i_batch in range(batch_size):
        for i_seq in range(seq_size):
            for i_joint in range(n_joints):
                u_p[i_joint] = joint_2d_pred[i_batch, i_seq, i_joint, 0]
                v_p[i_joint] = joint_2d_pred[i_batch, i_seq, i_joint, 1]
                u_t[i_joint] = joint_2d_target[i_batch, i_seq, i_joint, 0]
                v_t[i_joint] = joint_2d_target[i_batch, i_seq, i_joint, 1]
            if n_joints == 21:
                plt.plot([u_p[0], u_p[1]],[v_p[0], v_p[1]], 'm', linewidth=3)
                plt.plot([u_p[0], u_p[5]], [v_p[0], v_p[5]], 'm', linewidth=3)
                plt.plot([u_p[0], u_p[9]], [v_p[0], v_p[9]], 'm', linewidth=3)
                plt.plot([u_p[0], u_p[13]], [v_p[0], v_p[13]], 'm', linewidth=3)
                plt.plot([u_p[0], u_p[17]], [v_p[0], v_p[17]], 'm', linewidth=3)
                plt.plot([u_p[1], u_p[2]], [v_p[1], v_p[2]], 'm', linewidth=3)
                plt.plot([u_p[2], u_p[3]], [v_p[2], v_p[3]], 'm', linewidth=3)
                plt.plot([u_p[3], u_p[4]], [v_p[3], v_p[4]], 'm', linewidth=3)
                plt.plot([u_p[5], u_p[6]], [v_p[5], v_p[6]], 'm', linewidth=3)
                plt.plot([u_p[6], u_p[7]], [v_p[6], v_p[7]], 'm', linewidth=3)
                plt.plot([u_p[7], u_p[8]], [v_p[7], v_p[8]], 'm', linewidth=3)
                plt.plot([u_p[9], u_p[10]], [v_p[9], v_p[10]], 'm', linewidth=3)
                plt.plot([u_p[10], u_p[11]], [v_p[10], v_p[11]], 'm', linewidth=3)
                plt.plot([u_p[11], u_p[12]], [v_p[11], v_p[12]], 'm', linewidth=3)
                plt.plot([u_p[13], u_p[14]], [v_p[13], v_p[14]], 'm', linewidth=3)
                plt.plot([u_p[14], u_p[15]], [v_p[14], v_p[15]], 'm', linewidth=3)
                plt.plot([u_p[15], u_p[16]], [v_p[15], v_p[16]], 'm', linewidth=3)
                plt.plot([u_p[17], u_p[18]], [v_p[17], v_p[18]], 'm', linewidth=3)
                plt.plot([u_p[18], u_p[19]], [v_p[18], v_p[19]], 'm', linewidth=3)
                plt.plot([u_p[19], u_p[20]], [v_p[19], v_p[20]], 'm', linewidth=3)
                plt.imshow(image[i_batch, i_seq].cpu().permute(1, 2, 0))
                plt.savefig('%s/2D_skel/joint2D_%s_%03d_%03d_%02d_%02d.png' % (directory, mode, epoch + 1, i + 1, i_batch, i_seq))
                plt.clf()
            #
            # demo.display_hand({ 'verts': hand_verts[i_batch, i_seq, :, :].unsqueeze(0).cpu().detach(),
            #                     'joints': hand_joints[i_batch, i_seq, :, :].unsqueeze(0).cpu().detach()},mano_faces=mano_layer.th_faces)
            fig = plt.figure()
            # x = hand_verts[i_batch, i_seq, :, 0]
            # y = hand_verts[i_batch, i_seq, :, 1]
            # z = hand_verts[i_batch, i_seq, :, 2]
            # hand_verts[i_batch, i_seq, :, 0] = y
            # hand_verts[i_batch, i_seq, :, 1] = x
            # hand_verts[i_batch, i_seq, :, 2] = z
            ax = wrap_surfaces({'verts': hand_verts[i_batch, i_seq, :, :].unsqueeze(0).cpu().detach()* 1.2,
                                'joints': hand_joints[i_batch, i_seq, :, :].unsqueeze(0).cpu().detach() * 1.2},
                               mano_faces=mano_layer.th_faces, ax=fig)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_zticks([])
            ax.view_init(elev=16, azim=92)
            plt.axis('off')
            plt.savefig('%s/mesh1/mesh_%s_%03d_%03d_%02d_%02d_3D.png' % (directory, mode, epoch + 1, i + 1, i_batch, i_seq))
            # plt.show()
            plt.clf()


            fig = plt.figure()
            ax = wrap_surfaces({'verts': hand_verts[i_batch, i_seq, :, :].unsqueeze(0).cpu().detach() * 1.2 ,
                                'joints': hand_joints[i_batch, i_seq, :, :].unsqueeze(0).cpu().detach()* 1.2},
                               mano_faces=mano_layer.th_faces, ax=fig)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_zticks([])
            ax.view_init(elev=-48, azim=-101)
            plt.axis('off')
            plt.savefig(
                '%s/mesh2/mesh2_%s_%03d_%03d_%02d_%02d_3D.png' % (directory, mode, epoch + 1, i + 1, i_batch, i_seq))
            # plt.show()
            plt.clf()


            # ss = scale[i_batch, i_seq].cpu().numpy()
            # tu = trans[i_batch, i_seq, 0].cpu().numpy()
            # tv = trans[i_batch, i_seq, 1].cpu().numpy()
            # rot = rotation[i_batch, i_seq, :]
            # R = batch_rodrigues(rot.unsqueeze(0)).view(1, 3, 3)[0].cpu().numpy()
            # image_ = image[i_batch, i_seq].cpu().permute(1, 2, 0)
            # image_ = create_seq_synth_data(hand_verts[i_batch, i_seq, :, :], verts_2d[i_batch, i_seq, :, :],  color, mano_layer.th_faces.cpu().detach().numpy(),
            #                       ss, tu, tv, rot, 224, 224, image_)
            # fig = plt.figure()
            # plt.imshow(image_)
            # plt.savefig('%s/joint2D_%s_%03d_%03d_%02d_%02d_3D_overlap.png' % (directory, mode, epoch + 1, i + 1, i_batch, i_seq))
            # plt.show()




def hand_colors() :
       colors = []
       for i in range(0, 27):
              f = open('/home/vanyole/Research/HPE/SOTAmethods/[CVPR19]3dhand_inthewild/3dhand-master/data/meshes_colored/%d.obj' % i)
              cont = f.readlines()
              f.close()
              col = []
              for x in cont:
                     a = x[:len(x) - 1].split(" ")
                     if (a[0] == 'v'):
                            col.append(np.array([float(a[4]), float(a[5]), float(a[6])]))
              col = np.expand_dims(np.vstack(col), 0)
              colors.append(col)
       return np.vstack(colors)

def create_seq_synth_data(verts_3d, verts_2d, skin_color, f, ss, tu, tv, R, w, h, bg):
    import cv2

    rn = ColoredRenderer()
    ss = ss * 120.778145 + 753.188477 + 60
    tu = tu * 13.138042 + 94.337875 - 10
    tv = tv * 25.438021 + 92.918877 + 10
    bg = bg.detach().cpu().numpy()
    verts_3d = verts_3d.detach().cpu().numpy()
    verts_2d = verts_2d.detach().cpu().numpy()
    #
    # verts_3d_ = np.concatenate((verts_3d, np.ones((verts_3d.shape[0],1))), 1)
    # verts_2d_ = np.concatenate((verts_2d, np.ones((verts_2d.shape[0], 1))), 1)
    # # print(verts_3d_.shape, verts_2d_.shape)
    # verts_3d_inv = np.linalg.pinv(verts_3d_)
    # P = np.dot(verts_2d_.T, verts_3d_inv.T)
    # print(P.shape)
    # R = cv2.Rodrigues(rot)[0]
    # verts_3d = np.transpose(np.matmul(R, np.transpose(verts_3d))) / 1000.
    verts_3d = np.array([[ss, ss, 1], ] * 778) * verts_3d
    verts_3d = verts_3d + np.array([[tu, tv, 0], ] * 778)

    print( verts_2d , verts_3d)
    # verts_3d[:, 0] += tu
    # verts_3d[:, 1] += tv
    verts_3d[:, 2] = 10. + (verts_3d[:, 2] - np.mean(verts_3d[:, 2]))
    verts_3d[:, :2] = verts_3d[:, :2] * np.expand_dims(verts_3d[:, 2], 1)
    rn.camera = ProjectPoints(v=verts_3d, rt=np.zeros(3), t=np.array([0, 0, 0]), f=np.array([1, 1]),
                             c=np.array([0, 0]), k=np.zeros(5))

    rn.frustum = {'near': 1., 'far': 10000., 'width': w, 'height': h}
    rn.set(v=verts_3d, f=f, bgcolor=np.zeros(3))
    rn.vc = np.ones((778, 3))

    mask = rn.r.copy()
    mask = mask[:, :, 0].astype(np.uint8)

    rn.vc = skin_color
    hand = rn.r.copy() * 255.
    bg *=  255.
    # print(image.shape)
    print(bg.shape)
    image = (1 - np.expand_dims(mask, 2)) * bg + np.expand_dims(mask, 2) * hand

    # plt.scatter(verts_2d[:, 0], verts_2d[:, 1], c='r')
    # plt.scatter(verts_3d[:, 0], verts_3d[:, 1], c='b')
    # plt.imshow(image)
    # plt.show()
    image = image.astype(np.uint8)

    return image



def batch_rodrigues(axisang):
    #axisang N x 3
    axisang_norm = torch.norm(axisang + 1e-8, p=2, dim=1)
    angle = torch.unsqueeze(axisang_norm, -1)
    axisang_normalized = torch.div(axisang, angle)
    angle = angle * 0.5
    v_cos = torch.cos(angle)
    v_sin = torch.sin(angle)
    quat = torch.cat([v_cos, v_sin * axisang_normalized], dim=1)
    rot_mat = quat2mat(quat)
    rot_mat = rot_mat.view(rot_mat.shape[0], 9)
    return rot_mat

def displayMask(args, epoch, i, mask, predictions, mode):
    # Initialize directory
    directory = os.path.join('results', args.model_name + '_' + args.model_id, 'images')
    makeDir(directory)

    # Initialize predictions
    if args.model_name == 'EncoderConvLSTM_stoch':
        x2d_pred, x3d_pred, camera_param_pred, theta, beta, _, _ = predictions
    else : x2d_pred, x3d_pred, camera_param_pred, theta, beta = predictions

    batch_size, seq_size, _, _, _ = mask.size()
    y_hat = x2d_pred[:, :, 42:].view(batch_size, seq_size, 778, 2)

    # Save images
    batch_size = min(batch_size, 4)
    u = np.zeros(778)
    v = np.zeros(778)
    for i_batch in range(batch_size):
        for i_seq in range(seq_size):
            for ii in range(778):
                u[ii] = y_hat[i_batch, i_seq, ii, 0]
                v[ii] = y_hat[i_batch, i_seq, ii, 1]
            plt.plot(u, v, 'ro', markersize=1)
            plt.imshow(mask[i_batch, i_seq, 0:, :, :].cpu().permute(1, 2, 0))
            plt.savefig('%s/mask_%s_%03d_%03d_%02d_%02d.png' % (directory, mode, epoch+1, i+1, i_batch, i_seq))
            plt.clf()


def displayHand(args, epoch, i, predictions, targets, mode):
    """ Displays hand batch_idx in batch of hand_info, hand_info as returned by generate_random_hand """
    # Initialize directory
    directory = os.path.join('results', args.model_name + '_' + args.model_id, 'images')
    makeDir(directory)

    # Initialize ManoLayer
    mano_layerR = ManoLayer(mano_root=args.root_mano, use_pca=True, ncomps=10, flat_hand_mean=False, side='right')

    # Initialize predictions & targets
    if args.model_name == 'EncoderConvLSTM_stoch':
        _, x3d_pred, _, _, _, _, _ = predictions
    else : _, x3d_pred, _, _, _ = predictions

    batch_size, seq_size, _ = x3d_pred.size()
    joint_3d_pred = torch.stack((x3d_pred[:, :, :63:3], x3d_pred[:, :, 1:63:3], x3d_pred[:, :, 2:63:3]), dim=3)  # out2[:, :21, :]
    verts_3d_pred = x3d_pred[:, :, 63:].view(batch_size, seq_size, 778, 3)
    _, joint_3d_target, verts_3d_target, _, _ = targets

    # Initialize variables
    color_list = ['r', 'b']
    hand_info = [{'verts': verts_3d_pred.cpu(), 'joints': joint_3d_pred.cpu()},
                 {'verts': verts_3d_target.cpu(), 'joints': joint_3d_target.cpu()}]
    mano_faces = mano_layerR.th_faces
    ax = None
    alpha = 0.1

    # Save images
    batch_size = min(batch_size, 4)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for i_batch in range(batch_size):
        for i_seq in range(seq_size):
            for idx in range(len(hand_info)):
                verts, joints = hand_info[idx]['verts'][i_batch, i_seq], hand_info[idx]['joints'][i_batch, i_seq]
                if mano_faces is None:
                    ax.scatter(verts[:, 0], verts[:, 1], verts[:, 2], alpha=0.1)
                else:
                    mesh = Poly3DCollection(verts[mano_faces], alpha=alpha)
                    face_color = (141 / 255, 184 / 255, 226 / 255)
                    edge_color = (50 / 255, 50 / 255, 50 / 255)
                    mesh.set_edgecolor(edge_color)
                    mesh.set_facecolor(face_color)
                    ax.add_collection3d(mesh)
                ax.scatter(joints[:, 0], joints[:, 1], joints[:, 2], color=color_list[idx])
            cam_equal_aspect_3d(ax, verts.numpy())
            plt.savefig('%s/joint3D_%s_%03d_%03d_%02d_%02d.png' % (directory, mode, epoch+1, i+1, i_batch, i_seq))
            plt.cla()
    plt.close('all')


def cam_equal_aspect_3d(ax, verts, flip_x=False):
    """
    Centers view on cuboid containing hand and flips y and z axis
    and fixes azimuth
    """
    extents = np.stack([verts.min(0), verts.max(0)], axis=1)
    sz = extents[:, 1] - extents[:, 0]
    centers = np.mean(extents, axis=1)
    maxsize = max(abs(sz))
    r = maxsize / 2
    if flip_x:
        ax.set_xlim(centers[0] + r, centers[0] - r)
    else:
        ax.set_xlim(centers[0] - r, centers[0] + r)
    # Invert y and z axis
    ax.set_ylim(centers[1] + r, centers[1] - r)
    ax.set_zlim(centers[2] + r, centers[2] - r)



def quat2mat(quat):
    """Convert quaternion coefficients to rotation matrix.
    Args:
        quat: size = [batch_size, 4] 4 <===>(w, x, y, z)
    Returns:
        Rotation matrix corresponding to the quaternion -- size = [batch_size, 3, 3]
    """
    norm_quat = quat
    norm_quat = norm_quat / norm_quat.norm(p=2, dim=1, keepdim=True)
    w, x, y, z = norm_quat[:, 0], norm_quat[:, 1], norm_quat[:, 2], norm_quat[:, 3]

    batch_size = quat.size(0)

    w2, x2, y2, z2 = w.pow(2), x.pow(2), y.pow(2), z.pow(2)
    wx, wy, wz = w * x, w * y, w * z
    xy, xz, yz = x * y, x * z, y * z

    rotMat = torch.stack([
        w2 + x2 - y2 - z2, 2 * xy - 2 * wz, 2 * wy + 2 * xz, 2 * wz + 2 * xy,
        w2 - x2 + y2 - z2, 2 * yz - 2 * wx, 2 * xz - 2 * wy, 2 * wx + 2 * yz,
        w2 - x2 - y2 + z2], dim=1).view(batch_size, 3, 3)
    return rotMat


def adjustLR(optimizer, iters, base_lr, policy='step', policy_parameter=None, multiple=[1]):
    if policy == 'fixed':
        lr = base_lr
    elif policy == 'step':
        lr = base_lr * (policy_parameter['gamma'] ** (iters // policy_parameter['step_size']))
    elif policy == 'exp':
        lr = base_lr * (policy_parameter['gamma'] ** iters)
    elif policy == 'inv':
        lr = base_lr * ((1 + policy_parameter['gamma'] * iters) ** (-policy_parameter['power']))
    elif policy == 'multistep':
        lr = base_lr
        for stepvalue in policy_parameter['stepvalue']:
            if iters >= stepvalue:
                lr *= policy_parameter['gamma']
            else:
                break
    elif policy == 'poly':
        lr = base_lr * ((1 - iters * 1.0 / policy_parameter['max_iter']) ** policy_parameter['power'])
    elif policy == 'sigmoid':
        lr = base_lr * (1.0 / (1 + math.exp(-policy_parameter['gamma'] * (iters - policy_parameter['stepsize']))))
    elif policy == 'multistep-poly':
        lr = base_lr
        stepstart = 0
        stepend = policy_parameter['max_iter']
        for stepvalue in policy_parameter['stepvalue']:
            if iters >= stepvalue:
                lr *= policy_parameter['gamma']
                stepstart = stepvalue
            else:
                stepend = stepvalue
                break
        lr = max(lr * policy_parameter['gamma'], lr * (1 - (iters - stepstart) * 1.0 / (stepend - stepstart)) ** policy_parameter['power'])

    for i, param_group in enumerate(optimizer.param_groups):
        param_group['lr'] = lr * multiple[i]
    return lr


def makeDir(path):
    if not os.path.exists(path):
        print("Creating folder for %s..." % path)
        os.mkdir('/home/vanyole/Research/HPE/3dhand_sequence_20191022_findthebestframework/3dhand_sequence_20191022_untouched/'+ path)


def convertLossList(metrics, loss_list):
    metrics['loss_2d'].append(loss_list[0])
    metrics['loss_3d'].append(loss_list[1])
    metrics['loss_mask'].append(loss_list[2])
    metrics['loss_reg'].append(loss_list[3])
    metrics['loss_camera'].append(loss_list[4])
    # metrics['avg_distance_2d'] = np.concatenate([metrics['avg_distance_2d'], loss_list[5]], axis=0)
    # metrics['avg_distance_3d'] = np.concatenate([metrics['avg_distance_3d'], loss_list[6]], axis=0)
    n_kps = len(loss_list[5])
    [metrics['avg_distance_2d'][idx].append(loss_list[5][idx]) for idx in range(n_kps)]
    [metrics['avg_distance_3d'][idx].append(loss_list[6][idx]) for idx in range(n_kps)]
    return metrics


def setCUDA(args, data):
    image = data[0].to(device=args.device)
    mask = data[1].to(device=args.device)
    target_2d_joint = data[2].to(device=args.device)
    target_3d_joint = data[3].to(device=args.device)
    target_3d_verts = data[4].to(device=args.device)
    target_camera_param = data[5].to(device=args.device)
    dataset_type = data[6]
    index = data[7]
    return image, mask, (target_2d_joint, target_3d_joint, target_3d_verts, target_camera_param, dataset_type), index


def saveCheckpoint(args, model, optimizer, pretrain=False):
    directory = os.path.join('results', args.model_name + '_' + args.model_id)
    makeDir(directory)
    if pretrain:
        print("[Pretr] Saving checkpoint... %s" % directory)
        torch.save(model.state_dict(), os.path.join(directory, 'model_pretrain.pt'))
        torch.save(optimizer.state_dict(), os.path.join(directory, 'optimizer_pretrain.pt'))
        pickle.dump(args, open(os.path.join(directory, 'args_pretrain.pkl'), 'wb'))
    else:
        print("[Train] Saving checkpoint... %s" % directory)
        torch.save(model.state_dict(), os.path.join(directory, 'model.pt'))
        torch.save(optimizer.state_dict(), os.path.join(directory, 'optimizer.pt'))
        pickle.dump(args, open(os.path.join(directory, 'args.pkl'), 'wb'))


def saveCheckpointBestModel(args, model, optimizer, pretrain=False):
    directory = os.path.join('results', args.model_name + '_' + args.model_id)
    makeDir(directory)
    if pretrain:
        print("[Pretr] Saving best checkpoint... %s" % directory)
        torch.save(model.state_dict(), os.path.join(directory, 'model_pretrain_best.pt'))
        torch.save(optimizer.state_dict(), os.path.join(directory, 'optimizer_pretrain_best.pt'))
        pickle.dump(args, open(os.path.join(directory, 'args_pretrain_best.pkl'), 'wb'))
    else:
        print("[Train] Saving best checkpoint... %s" % directory)
        torch.save(model.state_dict(), os.path.join(directory, 'model_best.pt'))
        torch.save(optimizer.state_dict(), os.path.join(directory, 'optimizer_best.pt'))
        pickle.dump(args, open(os.path.join(directory, 'args_best.pkl'), 'wb'))

from collections import OrderedDict
def loadCheckpoint(args, model, optimizer, best=False, load_pretrain=True):
    directory = os.path.join('results', args.model_name + '_' + args.model_id)
    if best:
        if load_pretrain:
            print("[*****] Loading best pretrained checkpoint... %s" % directory)
            model_dict = torch.load(os.path.join(directory, 'model_pretrain_best.pt'))
            optimizer_dict = torch.load(os.path.join(directory, 'optimizer_pretrain_best.pt'))
            args = pickle.load(open(os.path.join(directory, 'args_pretrain_best.pkl'), 'rb'))
        else:
            print("[*****] Loading best checkpoint... %s" % directory)
            model_dict = torch.load(os.path.join(directory, 'model_best.pt'))
            optimizer_dict = torch.load(os.path.join(directory, 'optimizer_best.pt'))
            args = pickle.load(open(os.path.join(directory, 'args_best.pkl'), 'rb'))

    else:
        if load_pretrain:
            print("[*****] Loading pretrained checkpoint... %s" % directory)
            model_dict = torch.load(os.path.join(directory, 'model_pretrain.pt'))
            optimizer_dict = torch.load(os.path.join(directory, 'optimizer_pretrain.pt'))
            args = pickle.load(open(os.path.join(directory, 'args_pretrain.pkl'), 'rb'))
        else:
            print("[*****] Loading checkpoint... %s" % directory)
            model_dict = torch.load(os.path.join(directory, 'model.pt'))
            optimizer_dict = torch.load(os.path.join(directory, 'optimizer.pt'))
            args = pickle.load(open(os.path.join(directory, 'args.pkl'), 'rb'))
            # new_stat_dict = OrderedDict()
            # for key, value in list(model_dict.items()) :
            #     if key.split(".")[0] == "module":
            #         key_new = key[7:]
            #         new_stat_dict[key_new] = value
            # model_dict = new_stat_dict

    model.load_state_dict(model_dict)
    optimizer.load_state_dict(optimizer_dict)
    return args, model, optimizer

def displayPCK(args, directory, epoch, auc_2d, auc_3d, pck_curve_2d, pck_curve_3d, thresholds_2d, thresholds_3d, str):
    print(str)
    comparison_methods = {}
    if  'STEREO' in str :
        comparison_methods = STB_results
        y_l = 0.6
    elif 'EgoDexter' in str :
        comparison_methods = ED_results
        y_l = 0.2
    elif 'DexterObject' in str :
        comparison_methods = DO_results
        y_l = 0.2
    else :
        y_l = 0.



    curve_list_2d = []
    curve_list_2d.append((thresholds_2d, pck_curve_2d))
    curve_list_3d = []
    curve_list_3d.append((thresholds_3d, pck_curve_3d))
    fig = plt.figure()
    ax = fig.add_subplot(111)
    for t, v in curve_list_2d:
        ax.plot(t, v, label=args.model_name + ' (AUC=%.3f)' % auc_2d)
    ax.set_xlabel('Error Thresholds (px)')
    ax.set_ylabel('2D PCK')
    plt.ylim([0.0, 1.0])
    plt.grid()
    plt.legend(loc='lower right')
    plt.savefig('%s/PCK2D_%s_%03d.png' % (directory, str, epoch + 1))
    plt.clf()

    lw_ = 2


    fig = plt.figure()
    ax = fig.add_subplot(111)
    # file = open('%s/PCK3D_%s_%03d_plot.txt' % (directory, str, epoch + 1), 'w')
    np.save('%s/PCK3D_%s_%03d.npy' % (directory, str, epoch + 1), curve_list_3d)
    for t, v in curve_list_3d:
        #ax.plot(t*1000, v, label=args.model_name + ' (AUC=%.3f)' % auc_3d)
        ax.plot(t*1000, v, label='SeqHAND-NET (Ours)', linewidth=lw_)
        print(t,v)
    # file.write(curve_list_3d[1])
    # file.close()
    if comparison_methods :
        for method_name in list(comparison_methods.keys()) :
            if method_name == 'Boukhayma et al.(RGB/best)' :
                ax.plot(comparison_methods[method_name][:, 0], comparison_methods[method_name][:, 1], label=method_name,
                        linewidth=lw_, marker='o')
            elif method_name == 'Iqbal et al.' :
                ax.plot(comparison_methods[method_name][:, 0], comparison_methods[method_name][:, 1], label=method_name,
                        linewidth=lw_*1.5, marker='s', markevery=None)
            elif method_name == 'Cai et al.' :
                ax.plot(comparison_methods[method_name][:, 0], comparison_methods[method_name][:, 1], label=method_name,
                        linewidth=lw_*1.5, linestyle=':', color='r')
            else :
                ax.plot(comparison_methods[method_name][:, 0], comparison_methods[method_name][:, 1], label=method_name, linewidth=lw_)
    ax.set_xlabel('Error Threshold (mm)')
    ax.set_ylabel('3D PCK')
    plt.ylim([y_l, 1.0])
    plt.xlim([20.0, 50.0])
    plt.grid()
    plt.legend(loc='lower right')
    plt.savefig('%s/PCK3D_%s_%03d.png' % (directory, str, epoch + 1))
    plt.clf()


def saveLog(args, epoch, max_epochs, i, dataloader, learning_rate, loss, metrics, mode='Train'):
    data_n = type(dataloader.dataset).__name__
    """ https://github.com/spurra/vae-hands-3d/blob/master/evaluate_model.py """
    directory = os.path.join('results', args.model_name + '_' + args.model_id)
    makeDir(directory)
    mean_2d, median_2d, mean_3d, median_3d, auc_2d, auc_3d, pck_curve_2d, pck_curve_3d, thresholds_2d, thresholds_3d = getMeasures(args, metrics, data_n)  # [m] >> [mm]
    # if mode == 'Valid':
    #     directory_PCK = os.path.join('results', args.model_name + '_' + args.model_id, 'PCKs')
    #     makeDir(directory_PCK)
    #     str = mode + '_%s' % type(dataloader.dataset).__name__
    #     displayPCK(args, directory_PCK, epoch, auc_2d, auc_3d, pck_curve_2d, pck_curve_3d, thresholds_2d, thresholds_3d, str)
    with open(os.path.join(directory, 'log.txt'), 'a') as f:
        log = '[%s] Epoch:%3d/%3d Iter:%3d/%3d LR:%1.1e Loss:%5.2f(All) %5.2f(2D) %5.2f(3D) %5.2f(mask) %5.2f(reg) %5.2f(cam) DIST:%5.2fpx(%5.2fpx) %5.2fmm(%5.2fmm) AUC: %.3f(%.3f)' % \
              (mode, epoch + 1, max_epochs, i + 1, len(dataloader), learning_rate,
               np.mean(metrics['loss']), np.mean(metrics['loss_list']['loss_2d']), np.mean(metrics['loss_list']['loss_3d']), np.mean(metrics['loss_list']['loss_mask']), np.mean(metrics['loss_list']['loss_reg']), np.mean(metrics['loss_list']['loss_camera']),
               mean_2d, median_2d, mean_3d, median_3d, auc_2d, auc_3d)
        if mode == 'Valid':
            log = log + ', Dataset: %s' % type(dataloader.dataset).__name__
        print(log)
        f.write(log)
        f.write('\n')

def getMeasures(args, metrics, data_n):

    avg_distance_2d = metrics['loss_list']['avg_distance_2d']
    avg_distance_3d = metrics['loss_list']['avg_distance_3d']

    """ Outputs the average mean and median error as well as the pck score. """
    thresholds_2d = np.linspace(0.00, 30, 20)  # [px]
    thresholds_2d = np.array(thresholds_2d)
    norm_factor_2d = np.trapz(np.ones_like(thresholds_2d), thresholds_2d)
    thresholds_3d = np.linspace(0.02, 0.05, 20)  # [m]
    thresholds_3d = np.array(thresholds_3d)
    norm_factor_3d = np.trapz(np.ones_like(thresholds_3d), thresholds_3d)

    # init mean measures
    epe_mean_2d_all = list()
    epe_median_2d_all = list()
    epe_mean_3d_all = list()
    epe_median_3d_all = list()
    auc_2d_all = list()
    auc_3d_all = list()
    pck_curve_2d_all = list()
    pck_curve_3d_all = list()

    # Create one plot for each part
    if data_n == 'EgoDexter_seq' or data_n == 'DexterObject_seq':
        n_kps = 5
    else:
        n_kps = args.n_kps

    for part_id in range(n_kps):
        # mean/median error
        mean_2d, median_2d, mean_3d, median_3d = getEPE(avg_distance_2d[part_id], avg_distance_3d[part_id])
        if (mean_2d is None) or (mean_3d is None):
            continue  # there was no valid measurement for this keypoint
        epe_mean_2d_all.append(mean_2d)
        epe_median_2d_all.append(median_2d)
        epe_mean_3d_all.append(mean_3d)
        epe_median_3d_all.append(median_3d)

        # pck/auc
        pck_curve_2d = list()
        for t in thresholds_2d:
            pck_2d = getPCK(avg_distance_2d[part_id], t)
            pck_curve_2d.append(pck_2d)
        pck_curve_2d = np.array(pck_curve_2d)
        pck_curve_2d_all.append(pck_curve_2d)
        auc_2d = np.trapz(pck_curve_2d, thresholds_2d)
        auc_2d /= norm_factor_2d
        auc_2d_all.append(auc_2d)

        pck_curve_3d = list()
        for t in thresholds_3d:
            pck_3d = getPCK(avg_distance_3d[part_id], t)
            pck_curve_3d.append(pck_3d)
        pck_curve_3d = np.array(pck_curve_3d)
        pck_curve_3d_all.append(pck_curve_3d)
        auc_3d = np.trapz(pck_curve_3d, thresholds_3d)
        auc_3d /= norm_factor_3d
        auc_3d_all.append(auc_3d)

    epe_mean_2d = np.mean(np.array(epe_mean_2d_all))
    epe_median_2d = np.mean(np.array(epe_median_2d_all))
    epe_mean_3d = np.mean(np.array(epe_mean_3d_all))
    epe_median_3d = np.mean(np.array(epe_median_3d_all))
    auc_2d = np.mean(np.array(auc_2d_all))
    auc_3d = np.mean(np.array(auc_3d_all))
    pck_curve_2d = np.mean(np.array(pck_curve_2d_all), 0)  # mean only over keypoints
    pck_curve_3d = np.mean(np.array(pck_curve_3d_all), 0)  # mean only over keypoints

    return epe_mean_2d, epe_median_2d, epe_mean_3d*1000, epe_median_3d*1000, auc_2d, auc_3d, pck_curve_2d, pck_curve_3d, thresholds_2d, thresholds_3d

def getEPE(avg_distance_2d, avg_distance_3d):
    """ Returns end point error for one keypoint. https://github.com/spurra/vae-hands-3d/blob/master/utils/general.py """
    data_2d = np.concatenate(avg_distance_2d, axis=0)
    epe_mean_2d = np.mean(data_2d)
    epe_median_2d = np.median(data_2d)
    data_3d = np.concatenate(avg_distance_3d, axis=0)
    epe_mean_3d = np.mean(data_3d)
    epe_median_3d = np.median(data_3d)
    return epe_mean_2d, epe_median_2d, epe_mean_3d, epe_median_3d,

def getPCK(avg_distance_3d, threshold):
    """ Returns pck for one keypoint for the given threshold. https://github.com/spurra/vae-hands-3d/blob/master/utils/general.py """
    data = np.concatenate(avg_distance_3d, axis=0)
    # pck = np.mean((data[data!=0] <= threshold).astype('float'))
    pck = np.mean((data <= threshold).astype('float'))
    return pck

def calcAUC(x, y):
    """ Given x and y values it calculates the approx. integral and normalizes it: area under curve. https://github.com/spurra/vae-hands-3d/blob/master/utils/general.py """
    integral = np.trapz(y, x)
    norm = np.trapz(np.ones_like(y), x)
    return integral / norm
