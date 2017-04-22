import cv2
import numpy as np
from PIL import Image


class ImageCorrection:
    def __init__(self):
        pass

    @staticmethod
    def equalize_pil_from_cvmat(cvMat):
        return ImageCorrection.pilImToCv2Mat(
            ImageCorrection.equalize_pil(
                ImageCorrection.cv2MatToPilIm(cvMat)
            )
        )

    @staticmethod
    def equalize_cv2(cvMat):
        return cv2.equalizeHist(cvMat)

    @staticmethod
    def equalize_pil(im):
        layer = 1
        pix = im.getpixel((0, 0))
        if isinstance(pix, tuple):
            layer = len(pix)

        h = im.convert("L").histogram()
        lut = []
        for b in range(0, len(h), 256):
            # step size
            import operator
            step = max(reduce(operator.add, h[b:b + 256]) / 255, 1)
            # create equalization lookup table
            n = 0
            for i in range(256):
                lut.append(n / step)
                n = n + h[i + b]
        # map image through lookup table
        return im.point(lut * layer)

    @staticmethod
    def cv2MatToPilIm(cv2Mat):
        return Image.fromarray(cv2Mat)

    @staticmethod
    def pilImToCv2Mat(pilIm):
        return np.array(pilIm)

    @staticmethod
    def sharpenKernelCv2Mat(cv2Mat):
        kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
        return cv2.filter2D(cv2Mat, -1, kernel)

    @staticmethod
    def sharpenGaussianCv2Mat(cv2Mat):
        src2 = cv2.GaussianBlur(cv2Mat, (0, 0), 5)
        return cv2.addWeighted(cv2Mat, 1.8, src2, -0.8, 0)

    @staticmethod
    def cannyCv2Mat(cv2Mat):
        ret = cv2.Canny(cv2Mat, 50, 150, 3)
        return cv2.threshold(ret, 128, 255, cv2.THRESH_BINARY_INV)[1]

    @staticmethod
    def claheCv2Mat(cv2Mat):
        clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
        return clahe.apply(cv2Mat)

    @staticmethod
    def contrastCv2Mat(cv2Mat, value):
        pass

    @staticmethod
    def normalizeCv2Mat(cv2Mat):
        ret = None
        return cv2.normalize(cv2Mat, ret, alpha=20, beta=200, norm_type=cv2.NORM_MINMAX)

    @staticmethod
    def brightness(cv2Mat, beta):
        return cv2.add(cv2Mat, np.array([int(beta)], 'float64'))

    @staticmethod
    def contrast(cv2Mat, alpha):
        return cv2.multiply(cv2Mat, np.array([float(alpha)]))
