from pathlib import Path
import scipy.io as sio
from PIL import Image
import numpy as np
from utils.config import cfg
from data.base_obj import BaseObj
import random


class CmuObject(BaseObj):
    def __init__(self, sets, obj_resize):
        """
        :param sets: 'train' or 'test'
        :param obj_resize: resized object size
        """
        super(CmuObject, self).__init__()
        self.sets = sets
        self.classes = cfg.CMU.CLASSES
        self.kpt_len = [cfg.CMU.KPT_LEN for _ in cfg.CMU.CLASSES]

        self.root_path = Path(cfg.CMU.ROOT_DIR)
        self.obj_resize = obj_resize

        assert sets == 'train' or 'test', 'No match found for dataset {}'.format(sets)
        self.split_offset = cfg.CMU.TRAIN_OFFSET  # 0
        self.train_len = cfg.CMU.TRAIN_NUM

        self.fea_list = []
        if self.train_len == 5:
            for cls_name in self.classes:
                assert type(cls_name) is str
                cls_fea_list = [p for p in (self.root_path / cls_name).glob('*.scf')]
                cls_fea_list = sorted(cls_fea_list)
                ori_len = len(cls_fea_list)
                assert ori_len > 0, 'No data found for CMU Object Class. Is the dataset installed correctly?'
                if sets == 'train':
                    self.fea_list.append(
                            cls_fea_list[self.split_offset % ori_len: ori_len]
                        )
                else:
                    self.fea_list.append(
                            cls_fea_list[:self.split_offset % ori_len] +
                            cls_fea_list[(self.split_offset + self.train_len) % ori_len:]
                        )
        else:
            for cls_name in self.classes:
                assert type(cls_name) is str
                cls_fea_list = [p for p in (self.root_path / cls_name).glob('*.scf')]
                ori_len = len(cls_fea_list)
                assert ori_len > 0, 'No data found for CMU Object Class. Is the dataset installed correctly?'
                if self.split_offset % ori_len + self.train_len <= ori_len:
                    if sets == 'train':
                        self.fea_list.append(
                            cls_fea_list[self.split_offset % ori_len: (self.split_offset + self.train_len) % ori_len]
                        )
                    else:
                        self.fea_list.append(
                            cls_fea_list[:self.split_offset % ori_len] +
                            cls_fea_list[(self.split_offset + self.train_len) % ori_len:]
                        )
                else:
                    if sets == 'train':
                        self.fea_list.append(
                            cls_fea_list[:(self.split_offset + self.train_len) % ori_len - ori_len] +
                            cls_fea_list[self.split_offset % ori_len:]
                        )
                    else:
                        self.fea_list.append(
                            cls_fea_list[(self.split_offset + self.train_len) % ori_len - ori_len: self.split_offset % ori_len]
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
        if self.train_len == 5 and self.sets == 'train':
            for i in range(2):
                if (i == 0):
                    ran_int = random.randint(1,5)
                else:
                    ran_int = random.randint(6, len(self.fea_list[cls]))
                fea_name = self.fea_list[cls][ran_int-1]
                anno_dict = self.__get_anno_dict(fea_name, cls)
                if shuffle:
                    random.shuffle(anno_dict['fea'])
                anno_pair.append(anno_dict)
        else:
            # for i  in range(2):
            for fea_name in random.sample(self.fea_list[cls], 2):
                # fea_name = random.choice(self.fea_list[cls])
                anno_dict = self.__get_anno_dict(fea_name, cls)
                if shuffle:
                    random.shuffle(anno_dict['fea'])
                anno_pair.append(anno_dict)

        perm_mat = np.zeros_like(anno_dict['adj'], dtype=np.float32)
        row_list = []
        col_list = []
        for i, keypoint in enumerate(anno_pair[0]['fea']):
            for j, _keypoint in enumerate(anno_pair[1]['fea']):
                if keypoint['name'] == _keypoint['name']:
                    perm_mat[i, j] = 1
                    row_list.append(i)
                    col_list.append(j)
                    break
        row_list.sort()
        col_list.sort()
        perm_mat = perm_mat[row_list, :]
        perm_mat = perm_mat[:, col_list]
        anno_pair[0]['fea'] = [anno_pair[0]['fea'][i] for i in row_list]
        anno_pair[1]['fea'] = [anno_pair[1]['fea'][j] for j in col_list]

        return anno_pair, perm_mat
    
    def __get_anno_dict(self, fea_file, cls):
        """
        Get an feature from .scf file
        """
        assert fea_file.exists(), '{} does not exist.'.format(fea_file)

        adj_name = fea_file.stem + '.adj'
        adj_file = fea_file.parent / adj_name

        # file .scf
        lines_fea = open(fea_file).readlines()
        fea_list = []
        for idx, line in enumerate(lines_fea):
            attr = {'name': idx}
            attr['scf'] = np.array([float(p) for p in line.split()])
            fea_list.append(attr)

        # file .adj
        lines_adj = open(adj_file).readlines()
        adj_list = []
        for line in lines_adj:
            item = [float(p) for p in line.split()]
            adj_list.append(item)
        adj_ = np.array(adj_list)
        assert adj_.shape[0] == adj_.shape[1] == len(fea_list)

        anno_dict = dict()
        anno_dict['fea'] = fea_list
        anno_dict['adj'] = adj_
        anno_dict['cls'] = cls

        return anno_dict
