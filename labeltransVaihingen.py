from tqdm import tqdm
import numpy as np
import cv2
import os

# 标签中每个RGB颜色的值
VOC_COLORMAP = np.array([[0, 0, 0], [0, 0, 255], [0, 255, 255],
                         [0, 255, 0], [255, 255, 0]])
# 标签其标注的类别
VOC_CLASSES = ['impervious surfaces', 'buildings', 'low vegetation',
               'trees', 'cars']

# 处理txt中的对应图像
txt_path = r'E:\vencen\Project\Pycharm\CRGNet-main\dataset\vaihingen_test.txt'

# 标签所在的文件夹
label_file_path = r'E:\vencen\Project\Pycharm\CRGNet-main\Vaihingen\gt'
# 处理后的标签保存的地址
gray_save_path = r'E:\vencen\Project\Pycharm\CRGNet-main\Vaihingen\gt_new\\'

# txt_path = r'E:\vencen\Project\Pycharm\CRGNet-main\dataset\vaihingen_train_trans.txt'
# # 标签所在的文件夹
# label_file_path = r'E:\vencen\Project\Pycharm\CRGNet-main\Vaihingen\point\an1'
# # 处理后的标签保存的地址
# gray_save_path = r'E:\vencen\Project\Pycharm\CRGNet-main\Vaihingen\point_new\an1\\'

with open(txt_path, 'r') as f:
    file_names = f.readlines()
    for name in tqdm(file_names):
        name = name.strip('\n')  # 去掉换行符
        label_name = name   # label文件名
        label_url = os.path.join(label_file_path, label_name)

        mask = cv2.imread(label_url)
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB) # 通道转换
        mask = mask.astype(int)
        label_mask = np.zeros((mask.shape[0], mask.shape[1]), dtype=np.int16)
        # 标签处理
        for ii, label in enumerate(VOC_COLORMAP):
            locations = np.all(mask == label, axis=-1)
            label_mask[locations] = ii
        # 标签保存
        cv2.imwrite(gray_save_path+label_name, label_mask)
