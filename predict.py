
from fcn8 import FCN8
import glob
import cv2
import numpy as np
import random
import matplotlib.colors as mcolors

def getcenteroffset(shape, input_height, input_width):
    short_edge = min(shape[:2])
    xx = int((shape[0] - short_edge) / 2)
    yy = int((shape[1] - short_edge) / 2)
    return xx, yy

def image_enhance(img):
    img = img.astype(np.float32)
    # 减去Imagenet数据集的RGB均值，提高图像的识别率
    img[:, :, 0] -= 103.939
    img[:, :, 1] -= 116.779
    img[:, :, 2] -= 123.68
    return img

if __name__ == '__main__':
    n_classes = 15

    images_path = "dataset1/images_prepped_test/"
    segs_path = "dataset1/annotations_prepped_test/"

    input_height = 320
    input_width = 320

    # 类别配色表
    colors = mcolors.CSS4_COLORS
    colors_val = np.array([[int(c[1:3],16), int(c[3:5],16), int(c[5:7],16)] for c in colors.values()])
    np.random.shuffle(colors_val)


    images = sorted(glob.glob(images_path + "*.jpg") +
                    glob.glob(images_path + "*.png") +
                    glob.glob(images_path + "*.jpeg"))
    segmentations = sorted(glob.glob(segs_path + "*.jpg") +
                           glob.glob(segs_path + "*.png") +
                           glob.glob(segs_path + "*.jpeg"))

    # 创建模型并加载预训练权重
    model = FCN8 (n_classes, input_height, input_width)  # 有自定义层时，不能直接加载模型
    model.load_weights("output/fcn8_model.h5")

    for i, (imgName, segName) in enumerate(zip(images, segmentations)):

        print("%d/%d %s" % (i + 1, len(images), imgName))

        im = cv2.imread(imgName, 1)
        # 按指定大小截取测试图片中心部分
        xx, yy = getcenteroffset(im.shape, input_height, input_width)
        im = im[xx:xx + input_height, yy:yy + input_width, :]

        seg = cv2.imread(segName, 0)
        seg = seg[xx:xx + input_height, yy:yy + input_width]

        pr = model.predict(np.expand_dims(image_enhance(im), 0))[0]
        pr = pr.reshape((input_height, input_width, n_classes)).argmax(axis=2)

        cv2.imshow("img", im)
        pr_img = np.array([[colors_val[c] for c in row] for row in pr], dtype=np.uint8)

        cv2.imshow("seg_predict_res", pr_img)
        seg_img = np.array([[colors_val[c] for c in row] for row in seg], dtype=np.uint8)
        cv2.imshow("seg", seg_img)

        cv2.waitKey()