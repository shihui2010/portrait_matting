#import PIL
from PIL import Image
# from PIL.Image import EXTENSION
import random
import numpy as np
from crop_portrait import CascadeOpenCV
from itertools import product
import math
import scipy.misc

_PALETTE = [0, 0, 0,
           128, 0, 0,
           0, 128, 0,
           128, 128, 0,
           0, 0, 128,
           128, 0, 128,
           0, 128, 128,
           128, 128, 128,
           64, 0, 0,
           192, 0, 0,
           64, 128, 0,
           192, 128, 0,
           64, 0, 128,
           192, 0, 128,
           64, 128, 128,
           192, 128, 128,
           0, 64, 0,
           128, 64, 0,
           0, 192, 0,
           128, 192, 0,
           0, 64, 128,
           128, 64, 128,
           0, 192, 128,
           128, 192, 128,
           64, 64, 0,
           192, 64, 0,
           64, 192, 0,
           192, 192, 0]
_PAD = 50
MIN_IMG_SIZE = 120

FACE = CascadeOpenCV("haarcascade_frontalface_default.xml")


def blur_mask_edge(mask, pixel):
    """
    :param pixel: number of pixels for blurring at edge
    :param mask: "RGBA" format image
    :return: "RGBA" format blurred image
    """
    w, h = mask.size

    # identifying edge
    edge_points = []
    for x, y in product(range(w), range(h)):
        if mask.getpixel((x, y)) != (0, 0, 0, 255):
            continue
        xs = range(max(0, x - 1), min(w, x + 1))
        ys = range(max(0, y - 1), min(h, y + 1))
        for xi, yi in product(xs, ys):
            if mask.getpixel((xi, yi)) == (0, 0, 0, 0):
                edge_points.append((x, y))
                break

    # blur mask edge
    for x, y in edge_points:
        xs = range(max(0, x - pixel), min(w, x + pixel))
        ys = range(max(0, y - pixel), min(h, y + pixel))
        for xi, yi in product(xs, ys):
            min_dist = math.sqrt((x - xi) ** 2 + (y - yi) ** 2)
            offset = int(min_dist / float(pixel) * 255)
            origin = mask.getpixel((xi, yi))[3]
            if origin != 0 and offset < origin:
                mask.putpixel((xi, yi), (0, 0, 0, offset))
    return mask


def matting(img, label, bg, output_fname):
    img_label = np.asarray(label.convert(
            mode="P", palette=_PALETTE))
    assert img.shape[:2] == img_label.shape[:2], \
        "Label Image Size disagree with the One of Origin One"
    mat = np.where(img_label == 15)
    mask = Image.new("RGBA", img.shape[:2])
    for x, y in zip(mat[0].tolist(), mat[1].tolist()):
        mask.putpixel([y, x], (0, 0, 0, 255))

    mask = blur_mask_edge(mask, 10)

    # bg = Image.open(bg_fname)
    bw, bh = bg.size
    fw, fh = img.shape[:2]
    # if background smaller than fore-scene, then resize background
    if bw <= fw or bh <= fh:
        factor = max(float(fw) / bw, float(fh) / bh)
        bg = bg.resize((int(bw * factor + 10), int(bh * factor + 10)))
        bw, bh = bg.size
    sw = random.choice(range(0, bw - fw))
    sh = random.choice(range(0, bh - fh))
    bg_crop = bg.crop([sw, sh, sw + fw, sh + fh])
    bounding_box = [0, 0, fh, fw]
    # part = origin_img.crop(bounding_box)
    mask = mask.crop(bounding_box)
    # print part, bg_crop, mask
    # print img, bg_crop.size, mask.size
    # scipy.misc.toimage(img, cmin=0.0, cmax=256).save('output/test_0.jpg')
    img = Image.fromarray(img)
    # img.save("output/test.jpg")
    comp = Image.composite(img, bg_crop, mask)
    comp.save(output_fname)

if __name__ == "__main__":
    from crfasrnn_keras.src.crfrnn_model import get_crfrnn_model_def
    from crfasrnn_keras.src import util
    import sys
    import os
    import cv2

    # face cropper
    MAX_IMG_SIZE = 500
    PAD = 100
    FACE = CascadeOpenCV()

    bgs = list()
    for bg_fname in os.listdir("bg"):
        if bg_fname.endswith(".jpg"):
            bgs.append(Image.open(os.path.join("bg", bg_fname)))

    if len(sys.argv) == 1:
        data_dir = "big_figure"
    else:
        data_dir = sys.argv[1]

    model = get_crfrnn_model_def()

    saved_model_path = 'crfrnn_keras_model.h5'
    model.load_weights(saved_model_path)

    for fname in os.listdir(data_dir):
        input_file = os.path.join(data_dir, fname)

        img = cv2.imread(input_file, cv2.IMREAD_COLOR)
        # scipy.misc.toimage(img, cmin=0.0, cmax=256).save('output/test_origin.jpg')
        faces = FACE.run(img)
        if len(faces) != 1:
            continue
        img_size = img.shape
        x, y, w, h = faces[0]

        crop = img[
               max(y - PAD, 0):min(y + h + PAD, img_size[0]),
               max(x - PAD, 0):min(x + w + PAD, img_size[1])]
        if max(crop.shape) > MAX_IMG_SIZE:
            factor = float(MAX_IMG_SIZE) / max(crop.shape)
            crop = cv2.resize(crop, None,
                              fx=factor, fy=factor,
                              interpolation=cv2.INTER_CUBIC)
        img_data, img_h, img_w = util.get_preprocessed_image(crop)
        # img_data is np.array

        probs = model.predict(img_data, verbose=False)[0, :, :, :]
        label = util.get_label_image(probs, img_h, img_w)
        # label is np.array, segmentation of the img_data
        if not os.path.exists("output"):
            os.mkdir("output")

        crop = crop[...,::-1]

        for idx, bg in enumerate(bgs):
            out_fname = os.path.join("output",
                                     fname[:-4] + "_" + str(idx) + ".jpg")
            matting(crop, label, bg, out_fname)
