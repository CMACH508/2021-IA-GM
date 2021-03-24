from pathlib import Path
import scipy.io as sio
from PIL import Image, ImageDraw
import numpy as np
import math
import os

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

def draw_point(keys, img, fill_, r = 4):
    img = Image.fromarray(img)
    draw = ImageDraw.Draw(img)
    for i in range(len(keys)):
        y = keys[i][0]
        x = keys[i][1]
        draw.ellipse((x-r, y-r, x+r, y+r), fill=fill_)
    return img
    


def load_mat(file_path):
    f = sio.loadmat(file_path)
    img1, img2 = f['I1'], f['I2']
    # fea1, fea2 = f['features1'], f['features2']
    gt = f['gTruth']
    nums = gt.shape[1]
    fea1, fea2 = f['features1'][:nums], f['features2'][:nums]
    keys1_list = []
    keys2_list = []
    for k in range(nums):
        keys1_list.append(fea1[k][:2])
        keys2_list.append(fea2[k][:2])
    img1_key = draw_point(keys1_list, img1, (0, 0, 255))
    img2_key = draw_point(keys2_list, img2, (255, 0, 0))
    return img1_key, img2_key, nums

def test(cls_name):
    anno_path = 'anno_images/Motor'
    if not os.path.exists(anno_path):
        os.makedirs(anno_path)

    root_path = Path('pascal')
    pair_mat_list = [p for p in (root_path / cls_name).glob('*.mat')]
    pair_mat_list = sorted(pair_mat_list)
    for i in range(len(pair_mat_list)):
        mat_file = pair_mat_list[i]
        assert mat_file.exists(), '{} does not exist.'.format(mat_file)

        img1_key, img2_key, nums = load_mat(mat_file)

        for j in range(2):
            pair_name = mat_file.stem + '_' + str(j) + '.jpg'
            fname = os.path.join(anno_path, pair_name)
            if (j == 0):
                img1_key.save(fname)
            else:
                img2_key.save(fname)
        

def get_keypoints_mat(cls_name):
    root_path = Path('pascal')
    anno_path = 'PAC/' + cls_name
    pair_mat_list = [p for p in (root_path / cls_name).glob('*.mat')]
    pair_mat_list = sorted(pair_mat_list)
    for i in range(len(pair_mat_list)):
        mat_file = pair_mat_list[i]
        assert mat_file.exists(), '{} does not exist.'.format(mat_file)

        pair_data = sio.loadmat(mat_file)
        img1, img2 = pair_data['I1'], pair_data['I2']
        gt = pair_data['gTruth']
        nums = gt.shape[1]
        fea1, fea2 = pair_data['features1'][:nums], pair_data['features2'][:nums]
        keys1_list = []
        keys2_list = []
        print(nums)
        for k in range(nums):
            keys1_list.append((fea1[k][1], fea1[k][0]))
            keys2_list.append((fea2[k][1], fea2[k][0]))

        keys1 = np.array(keys1_list)
        keys2 = np.array(keys2_list)
        keys1 = np.transpose(keys1)
        keys2 = np.transpose(keys2)
        assert keys1.shape[0] == 2

        pair_name_a_mat = mat_file.stem + '_a.mat'
        fname1_mat = os.path.join(anno_path, pair_name_a_mat)
        sio.savemat(fname1_mat, {'pts_coord': keys1})
        pair_name_a_png = mat_file.stem + '_a.png'
        fname1_png = os.path.join(anno_path, pair_name_a_png)
        img_a = Image.fromarray(img1)
        img_a.save(fname1_png)


        pair_name_b = mat_file.stem + '_b.mat'
        fname2 = os.path.join(anno_path, pair_name_b)
        sio.savemat(fname2, {'pts_coord': keys2})
        pair_name_b_png = mat_file.stem + '_b.png'
        fname2_png = os.path.join(anno_path, pair_name_b_png)
        img_b = Image.fromarray(img2)
        img_b.save(fname2_png)

def key_images(cls_name):
    root_path = Path('PAC')
    anno_path = 'PAC/car_keys_imgs_10'
    if not os.path.exists(anno_path):
        os.makedirs(anno_path)

    mat_list = [p for p in (root_path / cls_name).glob('*.mat')]
    mat_list = sorted(mat_list)
    png_list = [p for p in (root_path / cls_name).glob('*.png')]
    png_list = sorted(png_list)

    for i in range(len(mat_list)):
        assert mat_list[i].stem == png_list[i].stem
        keys = sio.loadmat(mat_list[i])['pts_coord']

        img = Image.open(png_list[i])
        draw = ImageDraw.Draw(img)
        for j in range(keys.shape[1]):
            x = keys[0][j]
            y = keys[1][j]
            draw.ellipse((x-3, y-3, x+3, y+3), fill=(255, 0, 0))
        pair_name_b_png = png_list[i].stem + '.png'
        fname_png = os.path.join(anno_path, pair_name_b_png)
        img.save(fname_png)

def dist(a, b):
    x = a[0] - b[0]
    y = a[1] - b[1]
    return math.sqrt((x**2)+(y**2))


def shape_context(pairs):
    # pairs[i]:{x, y} size:n*2
    nums = int(pairs.shape[0])
    fea = np.zeros((nums, 60))

    alpha = 0
    for i in range(nums):
        for j in range(nums):
            alpha += dist(pairs[i], pairs[j])
    alpha /= (nums**2)

    for i in range(nums):
        for j in range(nums):
            if j == i:
                continue
            xco = pairs[j][0] - pairs[i][0]
            yco = pairs[j][1] - pairs[i][1]
            # convert to polar coordinates...

            theta = math.atan2(yco, xco) + math.pi
            r = dist(pairs[i], pairs[j])/alpha

            bin1, bin2 = 0, 0

            if theta < 2*math.pi/12:
                bin1 = 0
            elif theta < 4*math.pi/12:
                bin1 = 1
            elif theta < 6*math.pi/12:
                bin1 = 2
            elif theta < 8*math.pi/12:
                bin1 = 3
            elif theta < 10*math.pi/12:
                bin1 = 4
            elif theta < 12*math.pi/12:
                bin1 = 5
            elif theta < 14*math.pi/12:
                bin1 = 6
            elif theta < 16*math.pi/12:
                bin1 = 7
            elif theta < 18*math.pi/12:
                bin1 = 8
            elif theta < 20*math.pi/12:
                bin1 = 9
            elif theta < 22*math.pi/12:
                bin1 = 10
            else:
                bin1 = 11

            if r < 0.125:
                bin2 = 0
            elif r < 0.25:
                bin2 = 1
            elif r < 0.5:
                bin2 = 2
            elif r < 1:
                bin2 = 3
            elif r < 2:
                bin2 = 4
            else:
                continue
            
            fea[i][bin2*12 + bin1] += 1
    
    return fea

def get_scfs(cls_name):
    root_path = Path('PAC')
    anno_path = 'PAC/Motor_scf'
    if not os.path.exists(anno_path):
        os.makedirs(anno_path)

    mat_list = [p for p in (root_path / cls_name).glob('*.mat')]
    mat_list = sorted(mat_list)
    png_list = [p for p in (root_path / cls_name).glob('*.png')]
    png_list = sorted(png_list)

    for i in range(len(mat_list)):
        assert mat_list[i].stem == png_list[i].stem
        keys = sio.loadmat(mat_list[i])['pts_coord']
        keys = np.transpose(keys)
        assert keys.shape[1] == 2

        features = shape_context(pairs=keys)

        name_mat = mat_list[i].stem + '.mat'
        fname_mat = os.path.join(anno_path, name_mat)
        sio.savemat(fname_mat, {'fea': features})


if __name__ == "__main__":
    cls_name = 'Motor/'
    # test(cls_name)
    # get_keypoints_mat(cls_name)
    # key_images(cls_name)
    get_scfs(cls_name)