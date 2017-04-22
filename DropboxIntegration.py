import ntpath
import os
from multiprocessing import Process, Pipe
from threading import Thread

import dropbox
from dropbox.files import CommitInfo, WriteMode

from DatabaseStorage import DatabaseStorage


class DropboxIntegration:
    APP_KEY = 'cx2e3zv5qu44pqf'
    APP_SECRET = 'lmrjnwtqcct965x'

    STORING_FLOW = None

    def __init__(self):
        pass

    @staticmethod
    def feed_video_file_path_for_upload_async(path, mute=True, daemon=False):
        t = Thread(target=DropboxIntegration.__feed_video_upload_get_filename_thumb_url,
                   args=[path],
                   kwargs={'mute': mute})
        # t = Thread(target=DropboxIntegration.feed_video_file_path_for_upload, args=[path], kwargs={'mute': mute})
        t.daemon = daemon
        t.start()
        return t

    @staticmethod
    def __feed_video_upload_get_filename_thumb_url(path, mute=True, daemon=False):
        client = DropboxIntegration.get_dropbox_client()
        parent_conn, child_conn = Pipe()
        p = Process(target=DropboxIntegration.feed_video_file_path_for_upload2, args=[child_conn, client, path],
                    kwargs={'mute': mute})
        p.daemon = daemon
        p.start()
        result = parent_conn.recv()  # prints "[42, None, 'hello']"
        p.join()
        record, thumb, link = result
        print('Dropbox: write sqlite')
        DatabaseStorage.set_record(record, thumb, link)

    @staticmethod
    def get_dropbox_client():
        access_token = DatabaseStorage.get_dropbox_token()
        client = dropbox.Dropbox(access_token)
        return client

    @staticmethod
    def feed_video_file_path_for_upload(path, mute=True):
        filename = ntpath.basename(path)
        client = DropboxIntegration.get_dropbox_client()
        with open(path, 'rb') as f:
            file_size = os.path.getsize(path)

            CHUNK_SIZE = 4 * 1024 * 1024

            print('dpx: start upload %s' % path)
            if file_size <= CHUNK_SIZE:

                print client.files_upload(f, '/%s' % path)

            else:

                upload_session_start_result = client.files_upload_session_start(f.read(CHUNK_SIZE))
                cursor = dropbox.files.UploadSessionCursor(session_id=upload_session_start_result.session_id,
                                                           offset=f.tell())
                commit = dropbox.files.CommitInfo(path='/%s' % path)

                while f.tell() < file_size:
                    print('dpx: start upload %s (%d of %d)' % (path, f.tell(), file_size))
                    if (file_size - f.tell()) <= CHUNK_SIZE:
                        print client.files_upload_session_finish(f.read(CHUNK_SIZE),
                                                                 cursor,
                                                                 commit)
                    else:
                        client.files_upload_session_append(f.read(CHUNK_SIZE),
                                                           cursor.session_id,
                                                           cursor.offset)
                        cursor.offset = f.tell()
        print('Dropbox: upload %s ok' % path)
        print('Dropbox: thumb')
        base64_img = DropboxIntegration.get_base64_thumbnail_from_path('/%s' % filename)
        dropbox_url = "https://www.dropbox.com/home/Apps/PiSmartCam?preview=%s" % filename
        print('Dropbox: write sqlite')
        DatabaseStorage.set_record(filename.replace('.avi', ''), base64_img, dropbox_url)
        os.remove(path)

    @staticmethod
    def feed_video_file_path_for_upload2(conn, client, path, mute=True):
        filename = ntpath.basename(path)
        with open(path, 'rb') as f:
            file_size = os.path.getsize(path)

            CHUNK_SIZE = 4 * 1024 * 1024

            print('dpx: start upload %s' % path)
            if file_size <= CHUNK_SIZE:

                print client.files_upload(f, '/%s' % path)

            else:

                upload_session_start_result = client.files_upload_session_start(f.read(CHUNK_SIZE))
                cursor = dropbox.files.UploadSessionCursor(session_id=upload_session_start_result.session_id,
                                                           offset=f.tell())
                commit = dropbox.files.CommitInfo(path='/%s' % path)

                while f.tell() < file_size:
                    print('dpx: start upload %s (%d of %d)' % (path, f.tell(), file_size))
                    if (file_size - f.tell()) <= CHUNK_SIZE:
                        print client.files_upload_session_finish(f.read(CHUNK_SIZE),
                                                                 cursor,
                                                                 commit)
                    else:
                        client.files_upload_session_append(f.read(CHUNK_SIZE),
                                                           cursor.session_id,
                                                           cursor.offset)
                        cursor.offset = f.tell()
        print('Dropbox: upload %s ok' % path)
        print('Dropbox: thumb')
        base64_img = DropboxIntegration.get_base64_thumbnail_from_path('/%s' % filename)
        dropbox_url = "https://www.dropbox.com/home/Apps/PiSmartCam?preview=%s" % filename
        conn.send((filename.replace('.avi', ''), base64_img, dropbox_url))
        conn.close()
        os.remove(path)

    @staticmethod
    def get_base64_thumbnail_from_path(path):
        c = DropboxIntegration.get_dropbox_client()
        meta, r = c.files_get_thumbnail(path)
        import base64
        return 'data:image/png;base64,%s' % base64.b64encode(r.content)
