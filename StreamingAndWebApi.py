# http://www.chioka.in/python-live-video-streaming-example/
# http://pythonhosted.org/Flask-Classy/
# http://flask.pocoo.org/docs/0.12/quickstart/

# RESTful!

from Queue import Queue

import cv2
import numpy as np
from flask import Flask, Response, jsonify, send_file, abort, json, request, redirect
from datetime import timedelta
from flask import make_response, request, current_app
from functools import update_wrapper
from flask.ext.classy import FlaskView, route
import sqlite3
import os
import datetime
from DatabaseStorage import DatabaseStorage

# TODO: put into another thread to work
# TODO: sessions
from werkzeug.utils import secure_filename

from FacePreparation import FacePreparationDlib


def crossdomain(origin=None, methods=None, headers=None,
                max_age=21600, attach_to_all=True,
                automatic_options=True):
    if methods is not None:
        methods = ', '.join(sorted(x.upper() for x in methods))
    if headers is not None and not isinstance(headers, basestring):
        headers = ', '.join(x.upper() for x in headers)
    if not isinstance(origin, basestring):
        origin = ', '.join(origin)
    if isinstance(max_age, timedelta):
        max_age = max_age.total_seconds()

    def get_methods():
        if methods is not None:
            return methods

        options_resp = current_app.make_default_options_response()
        return options_resp.headers['allow']

    def decorator(f):
        def wrapped_function(*args, **kwargs):
            if automatic_options and request.method == 'OPTIONS':
                resp = current_app.make_default_options_response()
            else:
                resp = make_response(f(*args, **kwargs))
            if not attach_to_all and request.method != 'OPTIONS':
                return resp

            h = resp.headers

            h['Access-Control-Allow-Origin'] = origin
            h['Access-Control-Allow-Methods'] = get_methods()
            h['Access-Control-Max-Age'] = str(max_age)
            if headers is not None:
                h['Access-Control-Allow-Headers'] = headers
            return resp

        f.provide_automatic_options = False
        return update_wrapper(wrapped_function, f)

    return decorator


class StreamingAndWebApi:
    def __init__(self, debug=False):
        self.app = Flask(__name__)
        self.WebApiView.register(self.app, route_base='api')
        # self.WebApiView.register(self.app, route_base='/', subdomain='api')
        self.app.run(host='0.0.0.0', debug=debug)
        if debug:
            print('debug mode')

    class WebApiView(FlaskView):
        UPLOAD_TEMP_FOLDER = '/upload/temp'
        ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

        def __init__(self):
            super(StreamingAndWebApi.WebApiView, self).__init__()  # this is python 2 so ...
            self.__streamingBuffer = StreamingBuffer()

        @route('/stat')
        @crossdomain(origin='*')
        def stat(self):
            from sys import platform
            if platform == "linux" or platform == "linux2":
                # cat /proc/uptime
                import subprocess
                uptime_sec = float(subprocess.check_output(['cat', '/proc/uptime']).split(' ')[:1][0])
            else:
                uptime_sec = -3600.

            con = DatabaseStorage.get_connection()
            cursor = con.cursor()
            result = cursor.execute('SELECT '
                                    '(SELECT COUNT(*) FROM records) AS records_count, '
                                    '(SELECT COUNT(*) FROM faces) AS faces_count, '
                                    '(SELECT notifications FROM otr_counts LIMIT 1) AS noti_count').fetchone()
            con.close()
            ret = {
                'records_count': result[0],
                'faces_count': result[1],
                'notifications_count': result[2],
                'uptime': int(round(uptime_sec / 3600))
            }
            return jsonify(ret)

        @classmethod
        def allowed_file(cls, filename):
            return '.' in filename and filename.rsplit('.', 1)[1].lower() in cls.ALLOWED_EXTENSIONS

        @route('/')
        @crossdomain(origin='*')
        def main_test(self):
            return json.dumps({'status': 200, 'success': True}), 200, {'ContentType': 'application/json'}

        # video feed

        @route('/video_feed')
        @crossdomain(origin='*')
        def video_feed(self):
            return Response(self.__streamingBuffer.gen(),
                            mimetype='multipart/x-mixed-replace; boundary=frame')

        # face_list

        @route('/face', methods=['GET'])
        @crossdomain(origin='*')
        def list_faces(self):
            ret = {'faces': []}

            con = DatabaseStorage.get_connection()
            cursor = con.cursor()
            for row in cursor.execute('SELECT id, category, datetime '
                                      'FROM faces '
                                      'LEFT JOIN ('
                                      'SELECT datetime, face_id, MAX(rowid) FROM record_face GROUP BY face_id) '
                                      'AS [record_face] '
                                      'ON faces.id = record_face.face_id').fetchall():
                face_id = row[0]
                category = row[1].lower()
                file_dates = map(lambda x: os.path.getmtime(x[0]), os.walk("face/%s" % face_id))
                if len(file_dates) == 0:
                    last_update = None
                    img_path = None
                else:
                    # noinspection PyTypeChecker
                    last_update = datetime.datetime.utcfromtimestamp(max(file_dates))
                    img_path = [f for f in os.listdir('face/%s' % face_id)
                                if os.path.isfile(os.path.join('face/%s' % face_id, f))][0]
                last_detect = row[2]
                img_url = ''
                if img_path is not None:
                    img_url = '%s/%s/faces/%s' % (request.base_url, face_id, img_path)
                ret['faces'].append({
                    'id': face_id,
                    'category': category,
                    'last_update': last_update,
                    'last_detect': last_detect,
                    'img_url': img_url
                })
            con.close()
            return jsonify(ret)

        # face

        @route('/face', methods=['POST'])
        @crossdomain(origin='*')
        def new_face_from_name(self):
            payload = request.get_json(force=True, silent=True)
            if payload is None:
                return json.dumps({'status': 400, 'message': 'No payload supplied'}), 400, {
                    'ContentType': 'application/json'}

            for key in ['id', 'category']:
                if key not in payload:
                    return json.dumps({'status': 400, 'message': 'Invalid payload supplied'}), 400, {
                        'ContentType': 'application/json'}
            con = DatabaseStorage.get_connection()
            cursor = con.cursor()
            cursor.execute('INSERT INTO faces (id, category) VALUES (?, ?)', [payload['id'], payload['category'].lower()])
            con.commit()
            con.close()
            path = 'face/%s' % payload['id']
            os.mkdir(path)
            if cursor.rowcount < 1 or cursor.rowcount > 1:
                return json.dumps({'status': 500, 'message': 'Cannot insert into database.'}), 200, {
                    'ContentType': 'application/json'}
            return json.dumps({'status': 200, 'success': True}), 200, {'ContentType': 'application/json'}

        @route('/face/<name>', methods=['GET'])
        @crossdomain(origin='*')
        def get_face_from_name(self, name):
            con = DatabaseStorage.get_connection()
            cursor = con.cursor()
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
            category = result[1].lower()
            last_detect = result[2]
            faces = []
            file_dates = map(lambda x: os.path.getmtime(x[0]), os.walk("face/%s" % face_id))
            if len(file_dates) > 0:
                last_update = datetime.datetime.utcfromtimestamp(max(file_dates))
                for img_path in os.listdir('face/%s' % face_id):
                    if not os.path.isfile(os.path.join('face/%s' % face_id, img_path)):
                        continue
                    faces.append({
                        'filename': img_path,
                        'url': '%s/faces/%s' % (request.base_url, img_path)
                    })
            else:
                last_update = None

            detects = []
            for d in detects_dt:
                date = d[0]
                img = d[1]
                people = cursor.execute('SELECT face_id FROM record_face WHERE datetime=?', [date]).fetchall()
                detects.append({
                    'datetime': date,
                    'img': img,
                    'people': people
                })
            con.close()

            ret = {
                'id': face_id,
                'category': category,
                'lastUpdate': last_update,
                'lastDetect': last_detect,
                'faces': faces,
                'detects': detects
            }

            return jsonify(ret)

        @route('/face/<name>', methods=['POST'])
        @crossdomain(origin='*')
        def update_face_from_name(self, name):
            payload = request.get_json(force=True, silent=True)
            if payload is None:
                return json.dumps({'status': 400, 'message': 'No payload supplied'}), 400, {
                    'ContentType': 'application/json'}

            for key in ['id', 'category']:
                if key not in payload:
                    return json.dumps({'status': 400, 'message': 'Invalid payload supplied'}), 400, {
                        'ContentType': 'application/json'}

            con = DatabaseStorage.get_connection()
            cursor = con.cursor()
            cursor.execute('UPDATE faces SET id=?, category=? '
                           'WHERE id=?', [payload['id'], payload['category'].lower(), name])
            con.commit()
            con.close()
            oldPath = 'face/%s' % payload['id']
            newPath = 'face/%s' % payload['id']
            if oldPath != newPath:
                if os.path.isdir(oldPath):
                    os.rename(oldPath, newPath)
                else:
                    os.mkdir(newPath)
            if cursor.rowcount < 1:
                return json.dumps({'status': 404, 'message': 'Profile not found.'}), 404, {
                    'ContentType': 'application/json'}
            if cursor.rowcount > 1:
                return json.dumps({'status': 500, 'message': 'Unexpected database change.'}), 200, {
                    'ContentType': 'application/json'}
            return json.dumps({'status': 200, 'success': True}), 200, {'ContentType': 'application/json'}

        @route('/face/<name>', methods=['DELETE'])
        @crossdomain(origin='*')
        def delete_face_from_name(self, name):
            con = DatabaseStorage.get_connection()
            cursor = con.cursor()
            cursor.execute('DELETE FROM faces '
                           'WHERE id=?', [name])
            con.close()
            if cursor.rowcount < 1:
                return json.dumps({'status': 404, 'message': 'Profile not found.'}), 404, {
                    'ContentType': 'application/json'}
            if cursor.rowcount > 1:
                return json.dumps({'status': 500, 'message': 'Unexpected database change.'}), 200, {
                    'ContentType': 'application/json'}

            path = 'face/%s' % name

            try:
                import shutil
                shutil.rmtree(path, ignore_errors=True)
            except OSError as e:
                return json.dumps({'status': 500, 'message': e.strerror}), 500, {
                    'ContentType': 'application/json'}

            return json.dumps({'status': 200, 'success': True}), 200, {'ContentType': 'application/json'}

        # face_img

        @route('/face/<name>/faces/<pic_path>', methods=['GET'])
        @crossdomain(origin='*')
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
        @crossdomain(origin='*')
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
            path = 'uploads/%s' % name
            if not os.path.isdir(path):
                os.mkdir(path)
            path = os.path.join(path, filename)
            file.save(path)
            FacePreparationDlib().run_no_wait(path, 'face/%s' % name)
            return json.dumps({'status': 200, 'success': True}), 200, {'ContentType': 'application/json'}

        # face_list

        @route('/record', methods=['GET'])
        @crossdomain(origin='*')
        def list_records(self):
            ret = {'detects': []}

            con = DatabaseStorage.get_connection()
            cursor = con.cursor()
            for row in cursor.execute('SELECT datetime, img, `dropbox-url` FROM records').fetchall():
                datetime = row[0]
                img = row[1]
                dropbox_url = row[2]
                people = cursor.execute('SELECT face_id FROM record_face WHERE datetime=?', [datetime]).fetchall()
                ret['detects'].append({
                    'key': datetime,  # fix key problem in reactjs
                    'datetime': datetime,
                    'img': img,
                    'dropbox_url': dropbox_url,
                    'people': people
                })
            con.close()
            return jsonify(ret)

        # face_list

        @route('/record/<date_time>', methods=['GET'])
        @crossdomain(origin='*')
        def list_record_by_datetime(self, date_time):
            con = DatabaseStorage.get_connection()
            cursor = con.cursor()
            result = cursor.execute('SELECT datetime, img, `dropbox-url` FROM records WHERE datetime=?',
                                    [date_time]).fetchone()
            if result is None:
                return json.dumps({'status': 404, 'message': 'Requested profile not found.'}), 200, {
                    'ContentType': 'application/json'}
            datetime = result[0]
            img = result[1]
            dropbox_url = result[2]
            people = []
            for pe in cursor.execute('SELECT face_id, category '
                                     'FROM record_face '
                                     'LEFT JOIN faces ON faces.id = record_face.face_id '
                                     'WHERE datetime=?', [datetime]).fetchall():
                face_id = pe[0]
                category = pe[1].lower()
                img_path = [f for f in os.listdir('face/%s' % face_id)
                            if os.path.isfile(os.path.join('face/%s' % face_id, f))][0]
                people.append({
                    'face_id': face_id,
                    'category': category.lower(),
                    'img_url': '%s/face/%s/faces/%s' % (request.base_url.replace('/record/' + date_time, ''),
                                                        face_id, img_path)
                })
            con.close()
            return jsonify({
                'datetime': datetime,
                'img': img,
                'dropbox_url': dropbox_url,
                'people': people
            })

        # auth-login

        @route('/auth/login', methods=['POST'])
        @crossdomain(origin='*', methods='POST', headers='Content-Type')
        def auth_login(self):
            payload = request.get_json(force=True, silent=True)
            if payload is None:
                return json.dumps({'status': 400, 'message': 'No payload supplied'}), 400, {
                    'ContentType': 'application/json'}
            for key in ['uname', 'pwd']:
                if key not in payload:
                    return json.dumps({'status': 400, 'message': 'Invalid payload supplied'}), 400, {
                        'ContentType': 'application/json'}
            if not DatabaseStorage.check_login(payload['uname'], payload['pwd']):
                return json.dumps({'status': 401, 'message': 'Invalid login details.'}), 401, {
                    'ContentType': 'application/json'}

            from jose import jwt
            token = jwt.encode({'login': True}, 'secret', algorithm='HS256')
            return jsonify({
                'success': True,
                'jwt': token
            })

        # login

        @route('/set/login', methods=['GET'])
        @crossdomain(origin='*')
        def get_login_params(self):
            return jsonify(DatabaseStorage.get_login())

        @route('/set/login', methods=['POST'])
        @crossdomain(origin='*')
        def post_login_step2(self):
            payload = request.get_json(force=True, silent=True)
            if payload is None:
                return json.dumps({'status': 400, 'message': 'No payload supplied'}), 400, {
                    'ContentType': 'application/json'}
            for key in ['uname', 'pwd']:
                if key not in payload:
                    return json.dumps({'status': 400, 'message': 'Invalid payload supplied'}), 400, {
                        'ContentType': 'application/json'}
            con = DatabaseStorage.get_connection()
            cursor = con.cursor()
            cursor.execute('UPDATE settings SET uname=?, pwd=?',
                           [payload['uname'].strip(), payload['pwd'].strip()])
            con.commit()
            con.close()
            if cursor.rowcount < 1 or cursor.rowcount > 1:
                return json.dumps({'status': 500, 'message': 'Unexpected database state.'}), 200, {
                    'ContentType': 'application/json'}
            return json.dumps({'status': 200, 'success': True}), 200, {'ContentType': 'application/json'}

        # dropbox

        @route('/set/dropbox', methods=['GET'])
        @crossdomain(origin='*')
        def get_dropbox_params(self):
            return jsonify({
                'step1_url': '%s/step1' % request.base_url,
                'step1_url_redirect': '%s/step1/redirect' % request.base_url,
                'step2_url': '%s/step2' % request.base_url,
                'token': DatabaseStorage.get_dropbox_token()
            })

        @route('/set/dropbox/step1/redirect', methods=['GET'])
        @crossdomain(origin='*')
        def get_dropbox_step1_redirect(self):
            from dropbox import DropboxOAuth2FlowNoRedirect
            from DropboxIntegration import DropboxIntegration
            DropboxIntegration.STORING_FLOW = DropboxOAuth2FlowNoRedirect(DropboxIntegration.APP_KEY,
                                                                          DropboxIntegration.APP_SECRET)
            return redirect(DropboxIntegration.STORING_FLOW.start())

        @route('/set/dropbox/step1', methods=['GET'])
        @crossdomain(origin='*')
        def get_dropbox_step1(self):
            from dropbox import DropboxOAuth2FlowNoRedirect
            from DropboxIntegration import DropboxIntegration
            DropboxIntegration.STORING_FLOW = DropboxOAuth2FlowNoRedirect(DropboxIntegration.APP_KEY,
                                                                          DropboxIntegration.APP_SECRET)
            return jsonify({
                'url': DropboxIntegration.STORING_FLOW.start()
            })

        @route('/set/dropbox/step2', methods=['POST'])
        @crossdomain(origin='*')
        def post_dropbox_step2(self):
            from DropboxIntegration import DropboxIntegration
            if DropboxIntegration.STORING_FLOW is None:
                return json.dumps({'status': 400, 'message': 'You have to get the token in step 1 first '
                                                             'in order to make the flow works.'}), 400, {
                           'ContentType': 'application/json'}
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
                oauth_result = DropboxIntegration.STORING_FLOW.finish(payload['code'].strip())
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

        @route('/set/dropbox', methods=['POST'])
        @crossdomain(origin='*')
        def set_dropbox_params(self):
            payload = request.get_json(force=True, silent=True)
            if payload is None:
                return json.dumps({'status': 400, 'message': 'No payload supplied'}), 400, {
                    'ContentType': 'application/json'}
            for key in ['token']:
                if key not in payload:
                    return json.dumps({'status': 400, 'message': 'Invalid payload supplied'}), 400, {
                        'ContentType': 'application/json'}
            con = DatabaseStorage.get_connection()
            cursor = con.cursor()
            cursor.execute('UPDATE settings SET dropbox_token=?', [payload['token'].strip()])
            con.commit()
            con.close()
            if cursor.rowcount < 1 or cursor.rowcount > 1:
                return json.dumps({'status': 500, 'message': 'Unexpected database state.'}), 200, {
                    'ContentType': 'application/json'}
            return json.dumps({'status': 200, 'success': True}), 200, {'ContentType': 'application/json'}

        # gmail

        @route('/set/gmail', methods=['GET'])
        @crossdomain(origin='*')
        def get_gmail_params(self):
            return jsonify({
                'step1_url': '%s/step1' % request.base_url,
                'step1_url_redirect': '%s/step1/redirect' % request.base_url,
                'step2_url': '%s/step2' % request.base_url,
                'gmail_url_hostname': DatabaseStorage.get_gmail_url_hostname()
            })

        @route('/set/gmail/step1', methods=['GET'])
        @crossdomain(origin='*')
        def get_gmail_step1(self):
            from oauth2client import client
            from GmailIntegration import GmailIntegration
            GmailIntegration.STORING_FLOW = client.flow_from_clientsecrets(
                GmailIntegration.CLIENT_SECRET_FILE,
                GmailIntegration.SCOPES)
            GmailIntegration.STORING_FLOW.user_agent = 'PiSmartCamera'
            GmailIntegration.STORING_FLOW.redirect_uri = client.OOB_CALLBACK_URN
            return jsonify({
                'url': GmailIntegration.STORING_FLOW.step1_get_authorize_url()
            })

        @route('/set/gmail/step1/redirect', methods=['GET'])
        @crossdomain(origin='*')
        def get_gmail_step1(self):
            from oauth2client import client
            from GmailIntegration import GmailIntegration
            GmailIntegration.STORING_FLOW = client.flow_from_clientsecrets(GmailIntegration.CLIENT_SECRET_FILE,
                                                                           GmailIntegration.SCOPES)
            GmailIntegration.STORING_FLOW.user_agent = 'PiSmartCamera'
            GmailIntegration.STORING_FLOW.redirect_uri = client.OOB_CALLBACK_URN
            return redirect(GmailIntegration.STORING_FLOW.step1_get_authorize_url())

        @route('/set/gmail/step2', methods=['POST'])
        @crossdomain(origin='*')
        def post_gmail_step2(self):
            from GmailIntegration import GmailIntegration
            if GmailIntegration.STORING_FLOW is None:
                return json.dumps({'status': 400, 'message': 'You have to get the token in step 1 first '
                                                             'in order to make the flow works.'}), 400, {
                           'ContentType': 'application/json'}

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
                credentials = GmailIntegration.STORING_FLOW.step2_exchange(code=payload['code'].strip())
                store = Storage(GmailIntegration.TOKEN_FILE_PATH)
                store.put(credentials)
                # credentials.set_store(store)
                return json.dumps({'status': 200, 'success': True}), 200, {'ContentType': 'application/json'}
            except FlowExchangeError as e:
                return json.dumps({'status': 500, 'message': e.message}), 500, {
                    'ContentType': 'application/json'}

        @route('/set/gmail', methods=['POST'])
        @crossdomain(origin='*')
        def set_gmail_params(self):
            payload = request.get_json(force=True, silent=True)
            if payload is None:
                return json.dumps({'status': 400, 'message': 'No payload supplied'}), 400, {
                    'ContentType': 'application/json'}
            for key in ['gmail_url_hostname']:
                if key not in payload:
                    return json.dumps({'status': 400, 'message': 'Invalid payload supplied'}), 400, {
                        'ContentType': 'application/json'}
            con = DatabaseStorage.get_connection()
            cursor = con.cursor()
            cursor.execute('UPDATE settings SET gmail_url_hostname=?',
                           [payload['gmail_url_hostname'].strip()])
            con.commit()
            con.close()
            if cursor.rowcount < 1 or cursor.rowcount > 1:
                return json.dumps({'status': 500, 'message': 'Unexpected database state.'}), 200, {
                    'ContentType': 'application/json'}
            return json.dumps({'status': 200, 'success': True}), 200, {'ContentType': 'application/json'}

        # capture

        @route('/set/capture', methods=['GET'])
        @crossdomain(origin='*')
        def get_capture_params(self):
            result = DatabaseStorage.get_capture_params()
            if result is None:
                return json.dumps({'status': 500, 'message': 'Unexpected database state.'}), 200, {
                    'ContentType': 'application/json'}
            return jsonify(result)

        @route('/set/capture', methods=['POST'])
        @crossdomain(origin='*')
        def set_capture_params(self):
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
            con = DatabaseStorage.get_connection()
            cursor = con.cursor()
            cursor.execute('UPDATE settings SET capture_width=?, capture_height=?, capture_frame_rate=?, '
                           'process_width=?, process_height=?',
                           [payload['capture_width'],
                            payload['capture_height'],
                            payload['capture_frame_rate'],
                            payload['process_width'],
                            payload['process_height']])
            con.commit()
            con.close()
            if cursor.rowcount < 1 or cursor.rowcount > 1:
                return json.dumps({'status': 500, 'message': 'Unexpected database state.'}), 200, {
                    'ContentType': 'application/json'}
            return json.dumps({'status': 200, 'success': True}), 200, {'ContentType': 'application/json'}

        # motion

        @route('/set/motion', methods=['GET'])
        @crossdomain(origin='*')
        def get_motion_params(self):
            result = DatabaseStorage.get_motion_params()
            if result is None:
                return json.dumps({'status': 500, 'message': 'Unexpected database state.'}), 200, {
                    'ContentType': 'application/json'}
            return jsonify(result)

        @route('/set/motion', methods=['POST'])
        @crossdomain(origin='*')
        def set_motion_params(self):
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
            con = DatabaseStorage.get_connection()
            cursor = con.cursor()
            cursor.execute('UPDATE settings SET motion_threshold_low=?, '
                           'motion_minimum_area=?, '
                           'motion_bounding_box_padding=?,'
                           'motion_frame_span=?',
                           [payload['threshold_low'],
                            payload['minimum_area'],
                            payload['bounding_box_padding'],
                            payload['frame_span']])
            con.commit()
            con.close()
            if cursor.rowcount < 1 or cursor.rowcount > 1:
                return json.dumps({'status': 500, 'message': 'Unexpected database state.'}), 200, {
                    'ContentType': 'application/json'}
            return json.dumps({'status': 200, 'success': True}), 200, {'ContentType': 'application/json'}

        # face

        @route('/set/face', methods=['GET'])
        @crossdomain(origin='*')
        def get_face_params(self):
            result = DatabaseStorage.get_face_params()
            if result is None:
                return json.dumps({'status': 500, 'message': 'Unexpected database state.'}), 200, {
                    'ContentType': 'application/json'}
            return jsonify(result)

        @route('/set/face', methods=['POST'])
        @crossdomain(origin='*')
        def set_face_params(self):
            payload = request.get_json(force=True, silent=True)
            if payload is None:
                return json.dumps({'status': 400, 'message': 'No payload supplied'}), 400, {
                    'ContentType': 'application/json'}
            for key in ['face_method']:
                if key not in payload:
                    return json.dumps({'status': 400, 'message': 'Invalid payload supplied'}), 400, {
                        'ContentType': 'application/json'}
            con = DatabaseStorage.get_connection()
            cursor = con.cursor()
            cursor.execute('UPDATE settings SET face_method=?', [payload['face_method']])
            con.commit()
            con.close()
            if cursor.rowcount < 1 or cursor.rowcount > 1:
                return json.dumps({'status': 500, 'message': 'Unexpected database state.'}), 200, {
                    'ContentType': 'application/json'}
            return json.dumps({'status': 200, 'success': True}), 200, {'ContentType': 'application/json'}

        # facerec

        @route('/set/facerec', methods=['GET'])
        @crossdomain(origin='*')
        def get_facerec_params(self):
            result = DatabaseStorage.get_facerec_params()
            if result is None:
                return json.dumps({'status': 500, 'message': 'Unexpected database state.'}), 200, {
                    'ContentType': 'application/json'}
            return jsonify(result)

        @route('/set/facerec', methods=['POST'])
        @crossdomain(origin='*')
        def set_facerec_params(self):
            payload = request.get_json(force=True, silent=True)
            if payload is None:
                return json.dumps({'status': 400, 'message': 'No payload supplied'}), 400, {
                    'ContentType': 'application/json'}
            for key in ['facerec_method']:
                if key not in payload:
                    return json.dumps({'status': 400, 'message': 'Invalid payload supplied'}), 400, {
                        'ContentType': 'application/json'}
            con = DatabaseStorage.get_connection()
            cursor = con.cursor()
            cursor.execute('UPDATE settings SET facerec_method=?', [payload['facerec_method']])
            con.commit()
            con.close()
            if cursor.rowcount < 1 or cursor.rowcount > 1:
                return json.dumps({'status': 500, 'message': 'Unexpected database state.'}), 200, {
                    'ContentType': 'application/json'}
            return json.dumps({'status': 200, 'success': True}), 200, {'ContentType': 'application/json'}

        # record

        @route('/set/record', methods=['GET'])
        @crossdomain(origin='*')
        def get_record_params(self):
            result = DatabaseStorage.get_record_params()
            if result is None:
                return json.dumps({'status': 500, 'message': 'Unexpected database state.'}), 200, {
                    'ContentType': 'application/json'}
            return jsonify(result)

        @route('/set/record', methods=['POST'])
        @crossdomain(origin='*')
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
            con = DatabaseStorage.get_connection()
            cursor = con.cursor()
            cursor.execute('UPDATE settings SET record_width=?, record_height=?, record_framerate=?',
                           [payload['record_width'],
                            payload['record_height'],
                            payload['record_framerate']])
            con.commit()
            con.close()
            if cursor.rowcount < 1 or cursor.rowcount > 1:
                return json.dumps({'status': 500, 'message': 'Unexpected database state.'}), 200, {
                    'ContentType': 'application/json'}
            return json.dumps({'status': 200, 'success': True}), 200, {'ContentType': 'application/json'}

        # disk usage

        @route('/diskusage')
        @crossdomain(origin='*')
        def disk_usage(self):
            from sys import platform
            if platform == "linux" or platform == "linux2":
                import subprocess
                records_disks = subprocess.check_output(
                    ['df', '-m', '-x', 'tmpfs', '-x', 'devtmpfs', '--total']).strip().split('\n')
            else:
                records_disks = ['Filesystem     1M-blocks  Used Available Use% Mounted on',
                                 '/dev/root          29014  4707     22962  18% /',
                                 '/dev/mmcblk0p1        63    21        42  34% /boot',
                                 'total              29076  4727     23003  18% -']
            total = records_disks[-1].split()
            records_disks = records_disks[1:-1]
            disks = []
            for disk in records_disks:
                d = disk.split()
                disks.append({
                    'key': d[0],  # workaround nodejs bundle issue
                    'filesystem': d[0],
                    'total': ("%.2f" % (float(d[1]) / 1024)),
                    'used': ("%.2f" % (float(d[2]) / 1024)),
                    'available': ("%.2f" % (float(d[3]) / 1024)),
                    'use_percent': d[4],
                    'mounted_on': d[5]
                })
            ret = {
                'unit': 'GB',
                'total': ("%.2f" % (float(total[1]) / 1024)),
                'used': ("%.2f" % (float(total[2]) / 1024)),
                'available': ("%.2f" % (float(total[3]) / 1024)),
                'use_percent': total[4],
                'disks': disks
            }
            return jsonify(ret)

        # reinit

        @route('/set/reinit', methods=['POST'])
        @crossdomain(origin='*')
        def set_reinit_backend(self):
            func = request.environ.get('werkzeug.server.shutdown')
            if func is None:
                from ServiceEntryPoint import ServiceEntryPoint
                ServiceEntryPoint.API_REQUEST_REINIT = True
                raise ServerShutdown
            else:
                from ServiceEntryPoint import ServiceEntryPoint
                ServiceEntryPoint.API_REQUEST_REINIT = True
                func()
                return json.dumps({'status': 200, 'message': 'success'}), 200, {
                    'ContentType': 'application/json'}

        # reboot

        @route('/set/reboot', methods=['POST'])
        @crossdomain(origin='*')
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

        # trigger-alarm

        @route('/trigger/alarm', methods=['POST'])
        @crossdomain(origin='*')
        def set_trigger_alarm(self):
            payload = request.get_json(force=True, silent=True)
            if payload is None:
                return json.dumps({'status': 400, 'message': 'No payload supplied'}), 400, {
                    'ContentType': 'application/json'}
            for key in ['buzzing']:
                if key not in payload:
                    return json.dumps({'status': 400, 'message': 'Invalid payload supplied'}), 400, {
                        'ContentType': 'application/json'}

            # buzzing is boolean
            buzzing = bool(payload['buzzing'])
            from Alarming import Alarming
            Alarming.set_buzzing(buzzing)

            return json.dumps({'status': 200, 'success': True}), 200, {'ContentType': 'application/json'}

        # set/ir-filter

        @route('/set/ir-filter', methods=['get'])
        @crossdomain(origin='*')
        def get_ir_filter(self):
            from InfraRedLightFilter import InfraRedLightFilter
            return jsonify({
                'state': InfraRedLightFilter.get_state()
            })

        # @route('/dropbox_thumbnail', methods=['GET'])
        # @crossdomain(origin='*')
        # def video_feed(self):
        #     payload = request.get_json(force=True, silent=True)
        #     if payload is None:
        #         return json.dumps({'status': 400, 'message': 'No payload supplied'}), 400, {
        #             'ContentType': 'application/json'}
        #     for key in ['path']:
        #         if key not in payload:
        #             return json.dumps({'status': 400, 'message': 'Invalid payload supplied'}), 400, {
        #                 'ContentType': 'application/json'}
        #     from DropboxIntegration import DropboxIntegration
        #     meta, r = DropboxIntegration.get_dropbox_client().files_get_thumbnail(payload['path'])
        #     r.raw.decode_content = True
        #     return Response(r.raw, mimetype=r.headers['content-type'])


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


class ServerShutdown(Exception):
    pass
