import cv2

MIN_IMG_SIZE = 200


class CascadeOpenCV(object):
    def __init__(self, model_name="haarcascade_frontalface_default.xml",
                 min_size=5):
        self.min_size = min_size
        self.face_cascade = cv2.CascadeClassifier(model_name)

    def run(self, img):
        """
        :param img: cv2 imread object
        :return: list of bounding box [x, y, w, h]
        """
        if img is None:
            return None
        min_size = min(img.shape[:2])
        if min_size < MIN_IMG_SIZE:
            factor = MIN_IMG_SIZE / min_size
            img = cv2.resize(img, None, fx=factor, fy=factor, interpolation=cv2.INTER_CUBIC)

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray, 1.3, 5)
        return faces


if __name__ == "__main__":
    import os

    MAX_IMG_SIZE = 500
    PAD = 100
    FACE = CascadeOpenCV()
    path = "big_figure"
    output_idx = 0
    for fname in os.listdir(path):
        try:
            fname = os.path.join(path, fname)
            img = cv2.imread(fname)
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
            cv2.imwrite(path + "/face-" + str(output_idx) + ".jpg", crop)
            output_idx += 1
            print output_idx
        except:
            pass