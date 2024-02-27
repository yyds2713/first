import numpy as np
import cv2
import glob
import itertools
import matplotlib.pyplot as plt
import random
import tensorflow as tf
from tqdm.auto import tqdm

def generate_enhance(nClasses):

    def data_enhance(img, seg):

        img = img.astype(np.float32)
        # 减去Imagenet数据集的RGB均值，提高图像的识别率
        img[:, :, 0] -= 103.939
        img[:, :, 1] -= 116.779
        img[:, :, 2] -= 123.68

        seg_labels = np.zeros((img.shape[0], img.shape[1], nClasses))

        if seg is not None:
            # 生成带有分类维度的标记集合
            for c in range(nClasses):
                seg_labels[:, :, c] = (seg == c).astype(int)

            seg_labels = np.reshape(seg_labels, (-1, nClasses))
        else:
            seg_labels = None

        return img, seg_labels

    return data_enhance

def load_inputdata(images_path, segs_path):
    assert images_path[-1] == '/'  # assert 断言(后面指令没有错误或结果为False，程序终止并给出提示)
    assert segs_path[-1] == '/'

    images = sorted(glob.glob(images_path + "*.jpg") +
                    glob.glob(images_path + "*.png") + glob.glob(images_path + "*.jpeg"))

    segmentations = sorted(glob.glob(segs_path + "*.jpg") +
                           glob.glob(segs_path + "*.png") + glob.glob(segs_path + "*.jpeg"))

    return images, segmentations

def image_segmentation_generator(images, segmentations, batch_size, input_height, input_width, enhance=None):

    X,Y = [],[]
    # 格式: [('图像文件名','标注文件名'),('图像文件名','标注文件名'),..]
    # itertools.cycle 生成一个不断迭代的目标
    datas = itertools.cycle(zip(images, segmentations))
    # 样本数量
    data_size = len(images)

    idx = 0
    for im, seg in datas:
        # 使用opencv读取图像数据 [height, width, channel]  channel顺序(GBR)
        im = cv2.imread(im, 1)
        seg = cv2.imread(seg, 0)

        # 从原图中随机切割指定width,height的子图和标记
        yy = random.randint(0, im.shape[0] - input_height)
        xx = random.randint(0, im.shape[1] - input_width)

        im = im[yy:yy + input_height, xx:xx + input_width]
        seg = seg[yy:yy + input_height, xx:xx + input_width]

        # 使用应用数据强化
        im_h, seg_h = enhance(im, seg) if enhance else (im, seg)

        X.append(im_h)
        Y.append(seg_h)
        idx += 1

        if len(X) == batch_size:
            yield np.array(X), np.array(Y)
            X,Y = [],[]


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors

    colors = mcolors.CSS4_COLORS
    colors_val = np.array([[int(c[1:3],16), int(c[3:5],16), int(c[5:7],16)] for c in colors.values()])
    np.random.shuffle(colors_val)

    images_path = "dataset1/images_prepped_train/"
    segs_path = "dataset1/annotations_prepped_train/"

    """
    数据集加载并强化
    """
    images, segments = load_inputdata(images_path, segs_path)
    enhance_fn = generate_enhance(nClasses=15)
    gen = image_segmentation_generator(images, segments,
                                    batch_size=16, input_height=320, input_width=320,
                                    enhance=enhance_fn)
    
    img, seg = next(gen)
    print(img[0])
    print(seg[0])



    """
    数据集加载观察
    """
    images, segments = load_inputdata(images_path, segs_path)
    gen = image_segmentation_generator(images, segments, batch_size=16, input_height=320, input_width=320)
    img, seg = next(gen)

    # 查看原始图像和标记效果图
    seg_img = np.array([[colors_val[c] for c in row] for row in seg])
    plt.subplot(1,2,1)
    plt.imshow(img[10])
    plt.subplot(1,2,2)
    plt.imshow(seg_img[10])
    plt.show()