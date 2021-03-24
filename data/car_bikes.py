from pathlib import Path
import scipy.io as sio
from PIL import Image
import numpy as np
from utils.config import cfg
from utils.build_graphs import delaunay_triangulate
from data.base_obj import BaseObj
import random
import torch
from sklearn.preprocessing import scale, normalize


class PACObject(BaseObj):
    def __init__(self, sets, obj_resize):
        """
        :param sets: 'train' or 'test'
        :param obj_resize: resized object size
        """
        super(PACObject, self).__init__()
        self.sets = sets
        self.classes = cfg.PAC.CLASSES
        self.kpt_len = [cfg.PAC.KPT_LEN for _ in cfg.PAC.CLASSES]

        self.root_path = Path(cfg.PAC.ROOT_DIR)
        self.obj_resize = obj_resize

        assert sets == 'train' or 'test', 'No match found for dataset {}'.format(sets)
        self.split_offset = cfg.PAC.TRAIN_OFFSET  # 0
        self.train_len = cfg.PAC.TRAIN_NUM

        self.mat_list = []
        for cls_name in self.classes:
            assert type(cls_name) is str
            cls_mat_list = [p for p in (self.root_path / cls_name).glob('*.mat')]
            ori_len = len(cls_mat_list)
            assert ori_len > 0, 'No data found for WILLOW Object Class. Is the dataset installed correctly?'
            if self.split_offset % ori_len + self.train_len <= ori_len:
                if sets == 'train':
                    self.mat_list.append(
                        cls_mat_list[self.split_offset % ori_len: (self.split_offset + self.train_len) % ori_len]
                    )
                else:
                    self.mat_list.append(
                        cls_mat_list[:self.split_offset % ori_len] +
                        cls_mat_list[(self.split_offset + self.train_len) % ori_len:]
                    )
            else:
                if sets == 'train':
                    self.mat_list.append(
                        cls_mat_list[:(self.split_offset + self.train_len) % ori_len - ori_len] +
                        cls_mat_list[self.split_offset % ori_len:]
                    )
                else:
                    self.mat_list.append(
                        cls_mat_list[(self.split_offset + self.train_len) % ori_len - ori_len: self.split_offset % ori_len]
                    )

    def get_pair(self, cls=None, shuffle=True):
        """
        Randomly get a pair of objects from WILLOW-object dataset
        :param cls: None for random class, or specify for a certain set
        :param shuffle: random shuffle the keypoints
        :return: (pair of data, groundtruth permutation matrix)
        """
        if cls is None:
            cls = random.randrange(0, len(self.classes))
        elif type(cls) == str:
            cls = self.classes.index(cls)
        assert type(cls) == int and 0 <= cls < len(self.classes)

        anno_pair = []
        # for i  in range(2):
        # for fea_name in random.sample(self.mat_list[cls], 2):
        fea_name = random.choice(self.mat_list[cls])
        anno_pair = self.__get_anno_dict(fea_name, cls)
        for i  in range(2):
            if shuffle:
                random.shuffle(anno_pair[i]['keypoints_obj'])
            # anno_pair.append(anno_dict)

        perm_mat = np.zeros([len(_['keypoints_obj']) for _ in anno_pair], dtype=np.float32)
        assert len(anno_pair[0]['keypoints_obj']) == len(anno_pair[1]['keypoints_obj'])
        # perm_mat = np.zeros_like(anno_pair[0]['adj'], dtype=np.float32)
        row_list = []
        col_list = []
        for i, keypoint in enumerate(anno_pair[0]['keypoints_obj']):
            for j, _keypoint in enumerate(anno_pair[1]['keypoints_obj']):
                if keypoint['name'] == _keypoint['name']:
                    perm_mat[i, j] = 1
                    row_list.append(i)
                    col_list.append(j)
                    break
        row_list.sort()
        col_list.sort()
        perm_mat = perm_mat[row_list, :]
        perm_mat = perm_mat[:, col_list]
        anno_pair[0]['keypoints_obj'] = [anno_pair[0]['keypoints_obj'][i] for i in row_list]
        anno_pair[1]['keypoints_obj'] = [anno_pair[1]['keypoints_obj'][j] for j in col_list]

        return anno_pair, perm_mat
    
    def __get_anno_dict(self, mat_file, cls):
        """
        .mat file:
        # features1_out
        # BW1
        # edgeImage1
        # edgeComponents1
        # I1
        # features2_out
        # BW2
        # edgeImage2
        # edgeComponents2
        # I2
        # gTruth
        # features1
        # features2
        # nF1
        # nF2
        """
        assert mat_file.exists(), '{} does not exist.'.format(mat_file)

        pair_data = sio.loadmat(mat_file.open('rb'))
        img1_np, img2_np = pair_data['I1'], pair_data['I2']
        nums = pair_data['gTruth'].shape[1]
        fea1, fea2 = pair_data['features1'][:nums], pair_data['features2'][:nums]

        anno_pair = []
        anno_dict_1 = self.get_single_dict(img1_np, fea1, nums, cls)
        anno_dict_2 = self.get_single_dict(img2_np, fea2, nums, cls)
        anno_pair.append(anno_dict_1)
        anno_pair.append(anno_dict_2)

        return anno_pair

    def get_single_dict(self, img_np, fea, nums, cls):
        img = Image.fromarray(img_np)
        ori_size = img.size
        obj = img.resize(self.obj_resize, Image.ANTIALIAS) # (width, height)

        fea = scale(fea)

        fea_list = []
        keypoints_org_list = []
        keypoint_list = []
        for idx, keypoint in enumerate(fea):
            att = {'name': idx}
            att['scf'] = keypoint
            fea_list.append(att)

            attr = {'name': idx}
            attr['x'] = float(keypoint[1])
            attr['y'] = float(keypoint[0])
            keypoints_org_list.append(attr)

            attr_ = {'name': idx}
            attr_['x'] = float(keypoint[1]) * self.obj_resize[0] / ori_size[0]
            attr_['y'] = float(keypoint[0]) * self.obj_resize[1] / ori_size[1]
            keypoint_list.append(attr_)

        anno_dict = dict()
        anno_dict['fea'] = fea_list
        anno_dict['image'] = img_np
        anno_dict['keypoints'] = keypoints_org_list
        anno_dict['image_obj'] = obj
        anno_dict['keypoints_obj'] = keypoint_list
        anno_dict['cls'] = cls

        P_gt = [(kp['x'], kp['y']) for kp in anno_dict['keypoints']]

        P_gt = np.array(P_gt)
        adj = delaunay_triangulate(P_gt[0:nums, :])
        anno_dict['adj'] = adj

        return anno_dict

