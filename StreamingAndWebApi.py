# http://www.chioka.in/python-live-video-streaming-example/
# http://pythonhosted.org/Flask-Classy/
# http://flask.pocoo.org/docs/0.12/quickstart/

# RESTful!

from Queue import Queue

import cv2
import numpy as np
from flask import Flask, Response
from flask.ext.classy import FlaskView, route


# TODO: put into another thread to work
# TODO: sessions

class StreamingAndWebApi:
    def __init__(self):
        self.streamingBuffer = self.StreamingBuffer()

        self.app = Flask(__name__)
        self.WebApiView(self.streamingBuffer).register(self.app, route_base='/', subdomain='api')
        self.app.run()

    class WebApiView(FlaskView):
        def __init__(self, streamingBuffer):
            super(StreamingAndWebApi.WebApiView, self).__init__()  # this is python 2 so ...
            self.streamingBuffer = streamingBuffer

        @route('/video_feed')
        def video_feed(self):
            return Response(self.streamingBuffer.gen(),
                            mimetype='multipart/x-mixed-replace; boundary=frame')

        @route('/set/face/', methods=['GET', 'POST'])
        @route('/set/face/<name>/<int:picid>', methods=['GET', 'POST', 'PUT', 'DELETE'])
        def set_face(self, name=None, picid=None):
            # data: face: [
            #              {image base64?},
            #               ...
            #             ]
            pass

        @route('/set/facerec_params', methods=['GET', 'POST'])
        def set_facerec_params(self):
            pass

        @route('/set/dropbox', methods=['GET', 'POST'])
        def set_dropbox(self):
            pass

        @route('/set/gdrive', methods=['GET', 'POST'])
        def set_google_drive(self):
            pass

        @route('/load/facerec', methods=['GET', 'POST'])
        def reload_facerec_profile(self):
            pass

    class StreamingBuffer:
        def __init__(self):
            self.frame_queue = Queue()
            self.last_frame = self.encode(np.zeros((854, 480, 3), np.uint8))

        def encode(self, frame):
            return cv2.imencode('.jpg', frame)

        def gen(self):
            while True:
                if self.frame_queue.empty():
                    frame = self.last_frame
                else:
                    frame = self.encode(self.frame_queue.get())
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

        def putNewFrame(self, cv2Frame):
            self.frame_queue.put(cv2Frame)
