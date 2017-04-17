from threading import Thread

import cv2


class DataView:
    def __init__(self):
        pass

    @staticmethod
    def show_image(cv2mat, title=None, wait_key=True):
        if title is None:
            title = ""
        title = "Debug: %s" % title
        cv2.imshow(title, cv2mat)
        if wait_key:
            cv2.waitKey(0)
            cv2.destroyWindow(title)

    @staticmethod
    def show_image_nowait(cv2mat, title=None):
        t = Thread(target=DataView.show_image, args=[cv2mat], kwargs={"title": title})  # , "wait_key": False
        t.daemon = True
        t.start()