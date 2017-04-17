# http://www.chioka.in/python-live-video-streaming-example/
# http://pythonhosted.org/Flask-Classy/
# http://flask.pocoo.org/docs/0.12/quickstart/

# RESTful!

from Queue import Queue

import cv2
import numpy as np
from flask import Flask, Response, jsonify, send_file, abort, json, request
from flask.ext.classy import FlaskView, route
import sqlite3
import os
import datetime

# TODO: put into another thread to work
# TODO: sessions
from werkzeug.utils import secure_filename


class StreamingAndWebApi:
    def __init__(self):
        self.streamingBuffer = self.StreamingBuffer()

        self.app = Flask(__name__)
        self.WebApiView(self.streamingBuffer).register(self.app, route_base='/', subdomain='api')
        self.app.run()

    class WebApiView(FlaskView):
        UPLOAD_TEMP_FOLDER = '/upload/temp'
        ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])

        def __init__(self, streamingBuffer):
            self.__db = sqlite3.connect('pi_db.sqlite')

            self.__gmail_flow = None

            self.__dropbox_flow = None

            super(StreamingAndWebApi.WebApiView, self).__init__()  # this is python 2 so ...
            self.__streamingBuffer = streamingBuffer

        @classmethod
        def allowed_file(cls, filename):
            return '.' in filename and \
                   filename.rsplit('.', 1)[1].lower() in cls.ALLOWED_EXTENSIONS

        # video feed

        @route('/video_feed')
        def video_feed(self):
            return Response(self.__streamingBuffer.gen(),
                            mimetype='multipart/x-mixed-replace; boundary=frame')

        # disk usage

        @route('/diskusage')
        def disk_usage(self):
            import subprocess
            records_disks = subprocess.check_output(
                ['df', '-m', '-x', 'tmpfs', '-x', 'devtmpfs', '--total']).strip().split('\n')
            total = records_disks[-1].split()
            records_disks = records_disks[1:-1]
            disks = []
            for disk in records_disks:
                d = disk.split()
                disks.append({
                    'filesystem': d[0],
                    'total': d[1],
                    'used': d[2],
                    'available': d[3],
                    'use_percent': d[4],
                    'mounted_on': d[5]
                })
            ret = {
                'unit': 'MB',
                'total': total[1],
                'used': total[2],
                'available': total[3],
                'use_percent': total[4],
                'disks': disks
            }
            return jsonify(ret)

        # face_list

        @route('/face', methods=['GET'])
        def list_faces(self):
            ret = {'faces': []}

            cursor = self.__db.cursor()
            for row in cursor.execute('SELECT id, category, datetime '
                                      'FROM faces '
                                      'LEFT JOIN ('
                                      'SELECT datetime, face_id, MAX(rowid) FROM record_face GROUP BY face_id) '
                                      'AS [record_face] '
                                      'ON faces.id = record_face.face_id').fetchall():
                face_id = row[0]
                category = row[1]
                # noinspection PyTypeChecker
                last_update = datetime.datetime.utcfromtimestamp(
                    max(map(lambda x: os.path.getmtime(x[0]), os.walk("face/%s" % face_id)))
                )
                last_detect = row[2]
                img_path = os.listdir('face/%s' % face_id)[0]
                ret['faces'].append({
                    'id': face_id,
                    'category': category,
                    'last_update': last_update,
                    'last_detect': last_detect,
                    'img_url': '%s/face/%s/faces/%s' % (request.url_root, face_id, img_path)
                })
            return jsonify(ret)

        # face

        @route('/face/<name>', methods=['GET'])
        def get_face_from_name(self, name):
            cursor = self.__db.cursor()
            result = cursor.execute('SELECT id, category, datetime '
                                    'FROM faces '
                                    'LEFT JOIN ('
                                    'SELECT datetime, face_id, MAX(rowid) FROM record_face GROUP BY face_id) '
                                    'AS [record_face] '
                                    'ON faces.id = record_face.face_id '
                                    'WHERE id=?', [name]).fetchone()
            detects_dt = cursor.execute('SELECT records.datetime, img, `dropbox-url`, records.rowid '
                                        'FROM records '
                                        'INNER JOIN record_face ON record_face.datetime = records.datetime '
                                        'WHERE record_face.face_id=? '
                                        'ORDER BY records.rowid DESC', [name]).fetchall()
            if result is None:
                return json.dumps({'status': 404, 'message': 'Requested profile not found.'}), 200, {
                    'ContentType': 'application/json'}
            face_id = result[0]
            category = result[1]
            last_detect = result[2]
            # noinspection PyTypeChecker
            last_update = datetime.datetime.utcfromtimestamp(
                max(map(lambda x: os.path.getmtime(x[0]), os.walk("face/%s" % face_id)))
            )
            faces = []
            for img_path in os.listdir('face/charles'):
                faces.append({
                    'filename': img_path,
                    'url': '%s/face/%s/faces/%s' % (request.url_root, face_id, img_path)
                })

            detects = []
            for d in detects_dt:
                date = d[0]
                img = d[1]
                people = cursor.execute('SELECT face_id FROM record_face WHERE datetime=?', [date]).fetchall()
                detects.append({
                    'date': date,
                    'img': img,
                    'people': people
                })

            ret = {
                'id': face_id,
                'category': category,
                'lastUpdate': last_update,
                'lastDetect': last_detect,
                'faces': faces,
                'detects': detects
            }

            return jsonify(ret)

        @route('/face/<name>', methods=['PUT'])
        def update_face_from_name(self, name):
            payload = request.get_json(force=True, silent=True)
            if payload is None:
                return json.dumps({'status': 400, 'message': 'No payload supplied'}), 400, {
                    'ContentType': 'application/json'}

            for key in ['id', 'category']:
                if key not in payload:
                    return json.dumps({'status': 400, 'message': 'Invalid payload supplied'}), 400, {
                        'ContentType': 'application/json'}

            cursor = self.__db.cursor()
            cursor.execute('UPDATE faces SET id=?, category=? '
                           'WHERE id=?', [payload['id'], payload['category'], name])
            if cursor.rowcount < 1:
                return json.dumps({'status': 404, 'message': 'Profile not found.'}), 404, {
                    'ContentType': 'application/json'}
            if cursor.rowcount > 1:
                return json.dumps({'status': 500, 'message': 'Unexpected database change.'}), 200, {
                    'ContentType': 'application/json'}
            return json.dumps({'status': 200, 'success': True}), 200, {'ContentType': 'application/json'}

        @route('/face/<name>', methods=['DELETE'])
        def delete_face_from_name(self, name):
            cursor = self.__db.cursor()
            cursor.execute('DELETE FROM faces '
                           'WHERE id=?', [name])
            if cursor.rowcount < 1:
                return json.dumps({'status': 404, 'message': 'Profile not found.'}), 404, {
                    'ContentType': 'application/json'}
            if cursor.rowcount > 1:
                return json.dumps({'status': 500, 'message': 'Unexpected database change.'}), 200, {
                    'ContentType': 'application/json'}

            path = 'face/%s' % name

            try:
                import shutil
                shutil.rmtree(path)
            except OSError as e:
                return json.dumps({'status': 500, 'message': e.strerror}), 500, {
                    'ContentType': 'application/json'}

            return json.dumps({'status': 200, 'success': True}), 200, {'ContentType': 'application/json'}

        # face_img

        @route('/face/<name>/faces/<pic_path>', methods=['GET'])
        def get_face_img(self, name, pic_path):
            if '..' in pic_path or pic_path.startswith('/'):
                return json.dumps({'status': 404, 'message': 'File not found.'}), 404, {
                    'ContentType': 'application/json'}
            path = 'face/%s/%s' % (name, pic_path)
            if not os.path.isfile(path):
                return json.dumps({'status': 404, 'message': 'File not found.'}), 404, {
                    'ContentType': 'application/json'}
            return send_file(path)

        @route('/face/<name>/faces/<pic_path>', methods=['DELETE'])
        def del_face_img(self, name, pic_path):
            if '..' in pic_path or pic_path.startswith('/'):
                return json.dumps({'status': 404, 'message': 'File not found.'}), 404, {
                    'ContentType': 'application/json'}
            path = 'face/%s/%s' % (name, pic_path)
            if not os.path.isfile(path):
                return json.dumps({'status': 404, 'message': 'File not found.'}), 404, {
                    'ContentType': 'application/json'}
            try:
                os.remove(path)
                return json.dumps({'status': 200, 'success': True}), 200, {'ContentType': 'application/json'}
            except OSError as e:
                return json.dumps({'status': 500, 'message': e.strerror}), 500, {
                    'ContentType': 'application/json'}

        @route('/face/<name>/faces/', methods=['POST'])
        def upload_face_img(self, name):
            if 'file' not in request.files:
                return json.dumps({'status': 400, 'message': 'No file is supplied.'}), 400, {
                    'ContentType': 'application/json'}
            file = request.files['img']
            if file.filename == '':
                return json.dumps({'status': 400, 'message': 'No file is supplied.'}), 400, {
                    'ContentType': 'application/json'}
            if file and not self.allowed_file(file.filename):
                return json.dumps({'status': 406, 'message': 'File type not accepted (jpg/jpeg/gif/png).'}), 406, {
                    'ContentType': 'application/json'}
            filename = secure_filename(file.filename)
            path = 'face/%s' % name
            file.save(os.path.join(path, filename))
            return json.dumps({'status': 200, 'success': True}), 200, {'ContentType': 'application/json'}

        # dropbox

        @route('/set/gmail', methods=['GET'])
        def get_dropbox_params(self):
            return jsonify({
                'step1_url': '%s/set/gmail/step1' % request.url_root,
                'step2_url': '%s/set/gmail/step2' % request.url_root,
                'token': 'ranbu'
            })

        @route('/set/dropbox/step1', methods=['GET'])
        def get_dropbox_step1(self):
            from dropbox import DropboxOAuth2FlowNoRedirect
            from DropboxIntegration import DropboxIntegration
            self.__dropbox_flow = DropboxOAuth2FlowNoRedirect(DropboxIntegration.APP_KEY, DropboxIntegration.APP_SECRET)
            return jsonify({
                'url': self.__dropbox_flow.start()
            })

        @route('/set/dropbox/step2', methods=['POST'])
        def post_dropbox_step2(self):
            payload = request.get_json(force=True, silent=True)
            if payload is None:
                return json.dumps({'status': 400, 'message': 'No payload supplied'}), 400, {
                    'ContentType': 'application/json'}
            for key in ['code']:
                if key not in payload:
                    return json.dumps({'status': 400, 'message': 'Invalid payload supplied'}), 400, {
                        'ContentType': 'application/json'}
            from dropbox import oauth
            try:
                oauth_result = self.__dropbox_flow.finish(payload['code'].strip())
                return jsonify({
                    'token': oauth_result.access_token
                })
            except (oauth.BadRequestException,
                    oauth.BadStateException,
                    oauth.CsrfException,
                    oauth.NotApprovedException,
                    oauth.ProviderException) as e:
                return json.dumps({'status': 500, 'message': e.strerror}), 500, {
                    'ContentType': 'application/json'}

        # gmail

        @route('/set/gmail', methods=['GET'])
        def get_gmail_params(self):
            cursor = self.__db.cursor()
            result = cursor.execute('SELECT dropbox_token FROM settings').fetchone()
            return jsonify({
                'step1_url': '%s/set/gmail/step1' % request.url_root,
                'step2_url': '%s/set/gmail/step2' % request.url_root,
                'token': result[0]
            })

        @route('/set/gmail/step1', methods=['GET'])
        def get_gmail_step1(self):
            from oauth2client import client
            from GmailIntegration import GmailIntegration
            self.__gmail_flow = client.flow_from_clientsecrets(GmailIntegration.CLIENT_SECRET_FILE,
                                                               GmailIntegration.SCOPES)
            self.__gmail_flow.user_agent = 'PiSmartCamera'
            self.__gmail_flow.redirect_uri = client.OOB_CALLBACK_URN
            return jsonify({
                'url': self.__gmail_flow.step1_get_authorize_url()
            })

        @route('/set/gmail/step2', methods=['POST'])
        def post_gmail_step2(self):
            payload = request.get_json(force=True, silent=True)
            if payload is None:
                return json.dumps({'status': 400, 'message': 'No payload supplied'}), 400, {
                    'ContentType': 'application/json'}
            for key in ['code']:
                if key not in payload:
                    return json.dumps({'status': 400, 'message': 'Invalid payload supplied'}), 400, {
                        'ContentType': 'application/json'}
            from oauth2client.client import FlowExchangeError
            try:
                from oauth2client.file import Storage
                from GmailIntegration import GmailIntegration
                credentials = self.__gmail_flow.step2_exchange(code=payload['code'].strip())
                store = Storage(GmailIntegration.CLIENT_SECRET_FILE)
                store.put(credentials)
                # credentials.set_store(store)
                return json.dumps({'status': 200, 'success': True}), 200, {'ContentType': 'application/json'}
            except FlowExchangeError as e:
                return json.dumps({'status': 500, 'message': e.message}), 500, {
                    'ContentType': 'application/json'}

        @route('/set/gmail', methods=['POST'])
        def set_gmail_params(self):
            payload = request.get_json(force=True, silent=True)
            if payload is None:
                return json.dumps({'status': 400, 'message': 'No payload supplied'}), 400, {
                    'ContentType': 'application/json'}
            for key in ['face_method']:
                if key not in payload:
                    return json.dumps({'status': 400, 'message': 'Invalid payload supplied'}), 400, {
                        'ContentType': 'application/json'}
            cursor = self.__db.cursor()
            cursor.execute('UPDATE settings SET dropbox_token=?', [payload['dropbox_token'].split()]).fetchone()
            if cursor.rowcount < 1 or cursor.rowcount > 1:
                return json.dumps({'status': 500, 'message': 'Unexpected database state.'}), 200, {
                    'ContentType': 'application/json'}
            return json.dumps({'status': 200, 'success': True}), 200, {'ContentType': 'application/json'}

        # capture

        @route('/set/face', methods=['GET'])
        def get_face_params(self):
            cursor = self.__db.cursor()
            result = cursor.execute('SELECT capture_width, capture_height, capture_frame_rate, '
                                    'process_width, process_height FROM settings').fetchone()
            return jsonify({
                'capture': {
                    'width': result[0],
                    'height': result[1],
                    'frame_rate': result[2]
                },
                'process': {
                    'width': result[3],
                    'height': result[4]
                }
            })

        @route('/set/face', methods=['POST'])
        def set_face_params(self):
            payload = request.get_json(force=True, silent=True)
            if payload is None:
                return json.dumps({'status': 400, 'message': 'No payload supplied'}), 400, {
                    'ContentType': 'application/json'}
            for key in ['capture_width',
                        'capture_height',
                        'capture_frame_rate',
                        'process_width',
                        'process_height']:
                if key not in payload:
                    return json.dumps({'status': 400, 'message': 'Invalid payload supplied'}), 400, {
                        'ContentType': 'application/json'}
            cursor = self.__db.cursor()
            cursor.execute('UPDATE settings SET capture_width=?, capture_height=?, capture_frame_rate=?, '
                           'process_width=?, process_height=?',
                           [payload['capture_width'],
                            payload['capture_height'],
                            payload['capture_frame_rate'],
                            payload['process_width'],
                            payload['process_height']]).fetchone()
            if cursor.rowcount < 1 or cursor.rowcount > 1:
                return json.dumps({'status': 500, 'message': 'Unexpected database state.'}), 200, {
                    'ContentType': 'application/json'}
            return json.dumps({'status': 200, 'success': True}), 200, {'ContentType': 'application/json'}

        # motion

        @route('/set/motion', methods=['GET'])
        def get_face_params(self):
            cursor = self.__db.cursor()
            result = cursor.execute('SELECT motion_threshold_low, '
                                    'motion_minimum_area, '
                                    'motion_bounding_box_padding,'
                                    'motion_frame_span FROM settings').fetchone()
            if result is None:
                return json.dumps({'status': 500, 'message': 'Unexpected database state.'}), 200, {
                    'ContentType': 'application/json'}
            return jsonify({
                'threshold_low': result[0],
                'minimum_area': result[1],
                'bounding_box_padding': result[2],
                'frame_span': result[3]
            })

        @route('/set/motion', methods=['POST'])
        def set_face_params(self):
            payload = request.get_json(force=True, silent=True)
            if payload is None:
                return json.dumps({'status': 400, 'message': 'No payload supplied'}), 400, {
                    'ContentType': 'application/json'}
            for key in ['threshold_low',
                        'minimum_area',
                        'bounding_box_padding',
                        'frame_span']:
                if key not in payload:
                    return json.dumps({'status': 400, 'message': 'Invalid payload supplied'}), 400, {
                        'ContentType': 'application/json'}
            cursor = self.__db.cursor()
            cursor.execute('UPDATE settings SET motion_threshold_low=?, '
                           'motion_minimum_area=?, '
                           'motion_bounding_box_padding=?,'
                           'motion_frame_span=?',
                           [payload['threshold_low'],
                            payload['minimum_area'],
                            payload['bounding_box_padding'],
                            payload['frame_span']]).fetchone()
            if cursor.rowcount < 1 or cursor.rowcount > 1:
                return json.dumps({'status': 500, 'message': 'Unexpected database state.'}), 200, {
                    'ContentType': 'application/json'}
            return json.dumps({'status': 200, 'success': True}), 200, {'ContentType': 'application/json'}

        # face

        @route('/set/face', methods=['GET'])
        def get_face_params(self):
            cursor = self.__db.cursor()
            result = cursor.execute('SELECT face_method FROM settings').fetchone()
            if result is None:
                return json.dumps({'status': 500, 'message': 'Unexpected database state.'}), 200, {
                    'ContentType': 'application/json'}
            return jsonify({
                'face_method': result[0]
            })

        @route('/set/face', methods=['POST'])
        def set_face_params(self):
            payload = request.get_json(force=True, silent=True)
            if payload is None:
                return json.dumps({'status': 400, 'message': 'No payload supplied'}), 400, {
                    'ContentType': 'application/json'}
            for key in ['face_method']:
                if key not in payload:
                    return json.dumps({'status': 400, 'message': 'Invalid payload supplied'}), 400, {
                        'ContentType': 'application/json'}
            cursor = self.__db.cursor()
            cursor.execute('UPDATE settings SET face_method=?', [payload['face_method']]).fetchone()
            if cursor.rowcount < 1 or cursor.rowcount > 1:
                return json.dumps({'status': 500, 'message': 'Unexpected database state.'}), 200, {
                    'ContentType': 'application/json'}
            return json.dumps({'status': 200, 'success': True}), 200, {'ContentType': 'application/json'}

        # facerec

        @route('/set/facerec', methods=['GET'])
        def get_facerec_params(self):
            cursor = self.__db.cursor()
            result = cursor.execute('SELECT facerec_method FROM settings').fetchone()
            if result is None:
                return json.dumps({'status': 500, 'message': 'Unexpected database state.'}), 200, {
                    'ContentType': 'application/json'}
            return jsonify({
                'facerec_method': result[0]
            })

        @route('/set/facerec', methods=['POST'])
        def set_facerec_params(self):
            payload = request.get_json(force=True, silent=True)
            if payload is None:
                return json.dumps({'status': 400, 'message': 'No payload supplied'}), 400, {
                    'ContentType': 'application/json'}
            for key in ['facerec_method']:
                if key not in payload:
                    return json.dumps({'status': 400, 'message': 'Invalid payload supplied'}), 400, {
                        'ContentType': 'application/json'}
            cursor = self.__db.cursor()
            cursor.execute('UPDATE settings SET facerec_method=?', [payload['facerec_method']]).fetchone()
            if cursor.rowcount < 1 or cursor.rowcount > 1:
                return json.dumps({'status': 500, 'message': 'Unexpected database state.'}), 200, {
                    'ContentType': 'application/json'}
            return json.dumps({'status': 200, 'success': True}), 200, {'ContentType': 'application/json'}

        # record

        @route('/set/record', methods=['GET'])
        def get_record_params(self):
            cursor = self.__db.cursor()
            result = cursor.execute('SELECT record_width, record_height, record_framerate FROM settings').fetchone()
            if result is None:
                return json.dumps({'status': 500, 'message': 'Unexpected database state.'}), 200, {
                    'ContentType': 'application/json'}
            return jsonify({
                'record_width': result[0],
                'record_height': result[1],
                'record_framerate': result[2]
            })

        @route('/set/record', methods=['POST'])
        def set_record_params(self):
            payload = request.get_json(force=True, silent=True)
            if payload is None:
                return json.dumps({'status': 400, 'message': 'No payload supplied'}), 400, {
                    'ContentType': 'application/json'}
            for key in ['record_width',
                        'record_height',
                        'record_framerate']:
                if key not in payload:
                    return json.dumps({'status': 400, 'message': 'Invalid payload supplied'}), 400, {
                        'ContentType': 'application/json'}
            cursor = self.__db.cursor()
            cursor.execute('UPDATE settings SET record_width=?, record_height=?, record_framerate=?',
                           [payload['record_width'],
                            payload['record_height'],
                            payload['record_framerate']]).fetchone()
            if cursor.rowcount < 1 or cursor.rowcount > 1:
                return json.dumps({'status': 500, 'message': 'Unexpected database state.'}), 200, {
                    'ContentType': 'application/json'}
            return json.dumps({'status': 200, 'success': True}), 200, {'ContentType': 'application/json'}

        # reboot

        @route('/set/reinit', methods=['POST'])
        def set_reinit_backend(self):
            pass  # TODO: set_reinit_backend

        # reboot

        @route('/set/reboot', methods=['POST'])
        def set_reboot(self):
            try:
                FNULL = open(os.devnull, 'w')
                import subprocess
                retcode = subprocess.call(['sudo', 'reboot'], stdout=FNULL, stderr=FNULL)
                if retcode is not 0:
                    raise OSError(retcode, 'OSERROR')
                return json.dumps({'status': 200, 'success': True}), 200, {'ContentType': 'application/json'}
            except Exception as e:
                return json.dumps({'status': 500, 'message': e.message}), 200, {
                    'ContentType': 'application/json'}

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
                       b'Content-Type: image/jpeg\r\n\r\n' + frame.tobytes() + b'\r\n\r\n')

        def putNewFrame(self, cv2Frame):
            self.frame_queue.put(cv2Frame)
