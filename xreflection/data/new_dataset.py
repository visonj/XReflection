import math
import os.path
import os.path
import random
from os.path import join

import cv2
import numpy as np
import torch.utils.data
import torchvision.transforms.functional as TF
from PIL import Image
from xreflection.utils.registry import DATASET_REGISTRY
from scipy.signal import convolve2d
from scipy.stats import truncnorm
from .image_folder import make_dataset, read_fns
from .torchdata import Dataset as BaseDataset
from .transforms import to_tensor


def __scale_width(img, target_width):
    ow, oh = img.size
    if (ow == target_width):
        return img
    w = target_width
    h = int(target_width * oh / ow)
    h = math.ceil(h / 2.) * 2  # round up to even
    return img.resize((w, h), Image.BICUBIC)


def __scale_height(img, target_height):
    ow, oh = img.size
    if (oh == target_height):
        return img
    h = target_height
    w = int(target_height * ow / oh)
    w = math.ceil(w / 2.) * 2
    return img.resize((w, h), Image.BICUBIC)


def paired_data_transforms(img_1, img_2, unaligned_transforms=False):
    def get_params(img, output_size):
        w, h = img.size
        th, tw = output_size
        if w == tw and h == th:
            return 0, 0, h, w

        i = random.randint(0, h - th)
        j = random.randint(0, w - tw)
        return i, j, th, tw

    target_size = int(random.randint(384, 640) / 2.) * 2
    ow, oh = img_1.size
    if ow >= oh:
        img_1 = __scale_height(img_1, target_size)
        img_2 = __scale_height(img_2, target_size)
    else:
        img_1 = __scale_width(img_1, target_size)
        img_2 = __scale_width(img_2, target_size)

    if random.random() < 0.5:
        img_1 = TF.hflip(img_1)
        img_2 = TF.hflip(img_2)

    if random.random() < 0.5:
        angle = random.choice([90, 180, 270])
        img_1 = TF.rotate(img_1, angle)
        img_2 = TF.rotate(img_2, angle)

    i, j, h, w = get_params(img_1, (384, 384))
    img_1 = TF.crop(img_1, i, j, h, w)

    if unaligned_transforms:
        # print('random shift')
        i_shift = random.randint(-10, 10)
        j_shift = random.randint(-10, 10)
        i += i_shift
        j += j_shift

    img_2 = TF.crop(img_2, i, j, h, w)

    return img_1, img_2


class ReflectionSynthesis(object):
    def __init__(self):
        # Kernel Size of the Gaussian Blurry
        self.kernel_sizes = [5, 7, 9, 11]
        self.kernel_probs = [0.1, 0.2, 0.3, 0.4]

        # Sigma of the Gaussian Blurry
        self.sigma_range = [2, 5]
        self.alpha_range = [0.8, 1.0]
        self.beta_range = [0.4, 1.0]

    def __call__(self, T_, R_):
        T_ = np.asarray(T_, np.float32) / 255.
        R_ = np.asarray(R_, np.float32) / 255.

        kernel_size = np.random.choice(self.kernel_sizes, p=self.kernel_probs)
        sigma = np.random.uniform(self.sigma_range[0], self.sigma_range[1])
        kernel = cv2.getGaussianKernel(kernel_size, sigma)
        kernel2d = np.dot(kernel, kernel.T)
        for i in range(3):
            R_[..., i] = convolve2d(R_[..., i], kernel2d, mode='same')

        # a = np.random.uniform(self.alpha_range[0], self.alpha_range[1])
        # b = np.random.uniform(self.beta_range[0], self.beta_range[1])
        # T, R = a * T_, b * R_
        # 生成截断正态分布的参数
        a1 = truncnorm((0.82 - 1.109) / 0.118, (1.42 - 1.109) / 0.118, loc=1.109, scale=0.118)
        a2 = truncnorm((0.85 - 1.106) / 0.115, (1.35 - 1.106) / 0.115, loc=1.106, scale=0.115)
        a3 = truncnorm((0.85 - 1.078) / 0.116, (1.31 - 1.078) / 0.116, loc=1.078, scale=0.116)
        #print(T_.shape)
        b = np.random.uniform(self.beta_range[0], self.beta_range[1])
        T_[..., 0] *= a1.rvs()  # 对 T_ 的红色通道进行加权
        T_[..., 1] *= a2.rvs()  # 对 T_ 的绿色通道进行加权
        T_[..., 2] *= a3.rvs()  # 对 T_ 的蓝色通道进行加权
        T, R = T_, b * R_
        if random.random() < 0.7:
            I = T + R - T * R

        else:
            I = T + R
            if np.max(I) > 1:
                m = I[I > 1]
                m = (np.mean(m) - 1) * 1.3
                I = np.clip(T + np.clip(R - m, 0, 1), 0, 1)

        return T_, R_, I


class DataLoader(torch.utils.data.DataLoader):
    def __init__(self, dataset, batch_size, shuffle, *args, **kwargs):
        super(DataLoader, self).__init__(dataset, batch_size, shuffle, *args, **kwargs)
        self.shuffle = shuffle

    def reset(self):
        if self.shuffle:
            print('Reset Dataset...')
            self.dataset.reset()


@DATASET_REGISTRY.register()
class DSRDataset(BaseDataset):
    def __init__(self, opt):
        super(DSRDataset, self).__init__()
        self.size = opt['size']
        self.datadir = opt['datadir']
        self.enable_transforms = opt['enable_transforms']
        if opt['fns'] is None:
            self.fns = os.listdir(join(opt['datadir'], 'blended'))
        else:
            self.fns = read_fns(opt['fns'])
        sortkey = lambda key: os.path.split(key)[-1]
        self.paths = sorted(make_dataset(opt['datadir'], self.fns), key=sortkey)
        if self.size is not None:
            self.paths = np.random.choice(self.paths, self.size)

        self.syn_model = ReflectionSynthesis()
        self.reset(shuffle=False)

    def reset(self, shuffle=True):
        if shuffle:
            random.shuffle(self.paths)
        num_paths = len(self.paths) // 2
        self.B_paths = self.paths[0:num_paths]
        self.R_paths = self.paths[num_paths:2 * num_paths]

    def data_synthesis(self, t_img, r_img):
        if self.enable_transforms:
            t_img, r_img = paired_data_transforms(t_img, r_img)

        t_img, r_img, m_img = self.syn_model(t_img, r_img)

        B = TF.to_tensor(t_img)
        R = TF.to_tensor(r_img)
        M = TF.to_tensor(m_img)

        return B, R, M

    def __getitem__(self, index):
        index_B = index % len(self.B_paths)
        index_R = index % len(self.R_paths)

        B_path = self.B_paths[index_B]
        R_path = self.R_paths[index_R]

        t_img = Image.open(B_path).convert('RGB')
        r_img = Image.open(R_path).convert('RGB')

        B, R, M = self.data_synthesis(t_img, r_img)
        fn = os.path.basename(B_path)
        return {'input': M, 'target_t': B, 'target_r': M - B, 'fn': fn, 'real': False, 'inp_path': fn}

    def __len__(self):
        if self.size is not None:
            return min(max(len(self.B_paths), len(self.R_paths)), self.size)
        else:
            return max(len(self.B_paths), len(self.R_paths))


@DATASET_REGISTRY.register()
class DSRTestDataset(BaseDataset):
    def __init__(self, opt):
        super(DSRTestDataset, self).__init__()
        self.opt = opt
        self.size = opt['size']
        self.datadir = opt['datadir']
        self.mode = opt.get('mode', None)

        if opt['fns'] is None:
            self.fns = os.listdir(join(opt['datadir'], 'blended'))
        else:
            self.fns = read_fns(opt['fns'])

        self.enable_transforms = opt['enable_transforms']
        self.unaligned_transforms = opt['unaligned_transforms']
        self.round_factor = opt['round_factor']
        self.flag = opt['flag']
        self.if_align = True  # if_align

        if self.size is not None:
            self.fns = self.fns[:self.size]

    def align(self, x1, x2):
        h, w = x1.height, x1.width
        h, w = h // 32 * 32, w // 32 * 32
        x1 = x1.resize((w, h))
        x2 = x2.resize((w, h))
        return x1, x2

    def __getitem__(self, index):
        fn = self.fns[index]

        t_img = Image.open(join(self.datadir, 'transmission_layer', fn)).convert('RGB')
        m_img = Image.open(join(self.datadir, 'blended', fn)).convert('RGB')

        if self.if_align:
            t_img, m_img = self.align(t_img, m_img)

        if self.enable_transforms:
            t_img, m_img = paired_data_transforms(t_img, m_img, self.unaligned_transforms)

        if self.mode == "eval":
            w, h = t_img.size
            if max(w, h) > 512:
                if h > w:
                    t_img = t_img.resize((int(w * 512 / h), 512), Image.BILINEAR)
                    m_img = m_img.resize((int(w * 512 / h), 512), Image.BILINEAR)
                else:
                    t_img = t_img.resize((512, int(h * 512 / w)), Image.BILINEAR)
                    m_img = m_img.resize((512, int(h * 512 / w)), Image.BILINEAR)

        B = TF.to_tensor(t_img)
        M = TF.to_tensor(m_img)

        dic = {'input': M, 'target_t': B, 'fn': fn, 'real': True, 'target_r': M - B, 'inp_path': fn}
        if self.flag is not None:
            dic.update(self.flag)
        return dic

    def __len__(self):
        if self.size is not None:
            return min(len(self.fns), self.size)
        else:
            return len(self.fns)


class DSRTestDataset_zhu(BaseDataset):
    def __init__(self, datadir, fns=None, size=None, enable_transforms=False, unaligned_transforms=False,
                 round_factor=1, flag=None, if_align=True):
        super(DSRTestDataset_zhu, self).__init__()
        self.size = size
        self.datadir = datadir
        self.fns = fns or os.listdir(join(datadir, 'blended'))
        self.enable_transforms = enable_transforms
        self.unaligned_transforms = unaligned_transforms
        self.round_factor = round_factor
        self.flag = flag
        self.if_align = True  # if_align

        if size is not None:
            self.fns = self.fns[:size]

    def align(self, x1, x2):
        h, w = x1.height, x1.width
        h, w = h // 32 * 32, w // 32 * 32
        x1 = x1.resize((w, h))
        x2 = x2.resize((w, h))
        return x1, x2

    def __getitem__(self, index):
        fn = self.fns[index]

        t_img = Image.open(join(self.datadir, 'transmission_layer', fn)).convert('RGB')
        m_img = Image.open(join(self.datadir, 'blended', fn)).convert('RGB')

        if self.if_align:
            t_img, m_img = self.align(t_img, m_img)

        if self.enable_transforms:
            t_img, m_img = paired_data_transforms(t_img, m_img, self.unaligned_transforms)

        B = TF.to_tensor(t_img)
        M = TF.to_tensor(m_img)

        dic = {'input': M, 'target_t': B, 'fn': fn, 'real': True, 'target_r': M - B}
        if self.flag is not None:
            dic.update(self.flag)
        return dic

    def __len__(self):
        if self.size is not None:
            return min(len(self.fns), self.size)
        else:
            return len(self.fns)


class SIRTestDataset(BaseDataset):
    def __init__(self, datadir, fns=None, size=None, if_align=True):
        super(SIRTestDataset, self).__init__()
        self.size = size
        self.datadir = datadir
        self.fns = fns or os.listdir(join(datadir, 'blended'))
        self.if_align = if_align

        if size is not None:
            self.fns = self.fns[:size]

    def align(self, x1, x2, x3):
        h, w = x1.height, x1.width
        h, w = h // 32 * 32, w // 32 * 32
        x1 = x1.resize((w, h))
        x2 = x2.resize((w, h))
        x3 = x3.resize((w, h))
        return x1, x2, x3

    def __getitem__(self, index):
        fn = self.fns[index]

        t_img = Image.open(join(self.datadir, 'transmission_layer', fn)).convert('RGB')
        r_img = Image.open(join(self.datadir, 'reflection_layer', fn)).convert('RGB')
        m_img = Image.open(join(self.datadir, 'blended', fn)).convert('RGB')

        if self.if_align:
            t_img, r_img, m_img = self.align(t_img, r_img, m_img)

        B = TF.to_tensor(t_img)
        R = TF.to_tensor(r_img)
        M = TF.to_tensor(m_img)

        dic = {'input': M, 'target_t': B, 'fn': fn, 'real': True, 'target_r': R, 'target_r_hat': M - B}
        return dic

    def __len__(self):
        if self.size is not None:
            return min(len(self.fns), self.size)
        else:
            return len(self.fns)


class RealDataset(BaseDataset):
    def __init__(self, datadir, fns=None, size=None):
        super(RealDataset, self).__init__()
        self.size = size
        self.datadir = datadir
        self.fns = fns or os.listdir(join(datadir))

        if size is not None:
            self.fns = self.fns[:size]

    def align(self, x):
        h, w = x.height, x.width
        h, w = h // 32 * 32, w // 32 * 32
        x = x.resize((w, h))
        return x

    def __getitem__(self, index):
        fn = self.fns[index]
        B = -1
        m_img = Image.open(join(self.datadir, fn)).convert('RGB')
        M = to_tensor(self.align(m_img))
        data = {'input': M, 'target_t': B, 'fn': fn}
        return data

    def __len__(self):
        if self.size is not None:
            return min(len(self.fns), self.size)
        else:
            return len(self.fns)


@DATASET_REGISTRY.register()
class FusionDataset(BaseDataset):
    def __init__(self, opt):

        train_dataset_syn_opt = opt['train_dataset_syn']
        train_dataset_real_opt = opt['train_dataset_real']
        train_dataset_nature_opt = opt['train_dataset_nature']
        train_dataset_ntire_opt = opt['train_dataset_ntire']

        train_syn_dataset = DSRDataset(train_dataset_syn_opt)
        train_dataset_real = DSRTestDataset(train_dataset_real_opt)
        train_dataset_nature = DSRTestDataset(train_dataset_nature_opt)
        train_dataset_ntire = DSRTestDataset(train_dataset_ntire_opt)

        datasets = [train_syn_dataset, train_dataset_real, train_dataset_nature, train_dataset_ntire]

        self.datasets = datasets
        self.size = sum([len(dataset) for dataset in datasets])
        self.fusion_ratios = opt['fusion_ratios'] or [1. / len(datasets)] * len(datasets)
        print('[i] using a fusion dataset: %d %s imgs fused with ratio %s' % (
            self.size, [len(dataset) for dataset in datasets], self.fusion_ratios))

    def reset(self):
        for dataset in self.datasets:
            dataset.reset()

    def __getitem__(self, index):
        residual = 1
        for i, ratio in enumerate(self.fusion_ratios):
            if random.random() < ratio / residual or i == len(self.fusion_ratios) - 1:
                dataset = self.datasets[i]
                return dataset[index % len(dataset)]
            residual -= ratio

    def __len__(self):
        return self.size
