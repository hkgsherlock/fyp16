import cv2


class DataView:
    @staticmethod
    def show_image(cv2mat, title=None):
        if title is None:
            title = ""
        title = "Debug: %s" % title
        cv2.imshow(title, cv2mat)
        cv2.waitKey(0)
        cv2.destroyWindow(title)
