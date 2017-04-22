import sqlite3


class DatabaseStorage:
    def __init__(self):
        pass

    @classmethod
    def get_connection(cls):
        return sqlite3.connect('pi_db.sqlite')

    @classmethod
    def check_login(cls, uname, pwd):
        con = cls.get_connection()
        cursor = con.cursor()
        result = cursor.execute('SELECT 1 FROM settings WHERE uname=? AND pwd=?', [uname, pwd]).fetchone()
        con.close()
        return result is not None

    @classmethod
    def get_login(cls):
        con = cls.get_connection()
        cursor = con.cursor()
        result = cursor.execute('SELECT uname, pwd FROM settings').fetchone()
        con.close()
        return {
            'uname': result[0],
            'pwd': result[1]
        }

    @classmethod
    def get_dropbox_token(cls):
        con = cls.get_connection()
        cursor = con.cursor()
        result = cursor.execute('SELECT dropbox_token FROM settings').fetchone()
        con.close()
        return result[0]

    @classmethod
    def get_gmail_url_hostname(cls):
        con = cls.get_connection()
        cursor = con.cursor()
        result = cursor.execute('SELECT gmail_url_hostname FROM settings').fetchone()
        con.close()
        return result[0]

    @classmethod
    def get_capture_params(cls):
        con = cls.get_connection()
        cursor = con.cursor()
        result = cursor.execute('SELECT capture_width, capture_height, capture_frame_rate, '
                                'process_width, process_height FROM settings').fetchone()
        con.close()
        if result is None:
            return None
        return {
            'capture': {
                'width': result[0],
                'height': result[1],
                'frame_rate': result[2]
            },
            'process': {
                'width': result[3],
                'height': result[4]
            }
        }

    @classmethod
    def get_motion_params(cls):
        con = cls.get_connection()
        cursor = con.cursor()
        result = cursor.execute('SELECT motion_threshold_low, '
                                'motion_minimum_area, '
                                'motion_bounding_box_padding,'
                                'motion_frame_span FROM settings').fetchone()
        con.close()
        if result is None:
            return None
        return {
            'threshold_low': result[0],
            'minimum_area': result[1],
            'bounding_box_padding': result[2],
            'frame_span': result[3]
        }

    @classmethod
    def get_face_params(cls):
        con = cls.get_connection()
        cursor = con.cursor()
        result = cursor.execute('SELECT face_method FROM settings').fetchone()
        con.close()
        if result is None:
            return None
        return {
            'face_method': result[0]
        }

    @classmethod
    def get_facerec_params(cls):
        con = cls.get_connection()
        cursor = con.cursor()
        result = cursor.execute('SELECT facerec_method FROM settings').fetchone()
        con.close()
        if result is None:
            return None
        return {
            'facerec_method': result[0]
        }

    @classmethod
    def get_record_params(cls):
        con = cls.get_connection()
        cursor = con.cursor()
        result = cursor.execute('SELECT record_width, record_height, record_framerate FROM settings').fetchone()
        con.close()
        if result is None:
            return None
        return {
            'record_width': result[0],
            'record_height': result[1],
            'record_framerate': result[2]
        }

    @classmethod
    def set_record_face(cls, datetime, faceid):
        con = cls.get_connection()
        cursor = con.cursor()
        cursor.execute('INSERT INTO record_face (datetime, face_id) VALUES (?, ?)', [datetime, faceid])
        con.commit()
        con.close()
        return cursor.rowcount == 0

    @classmethod
    def set_record(cls, datetime, img, dropbox_url):
        con = cls.get_connection()
        cursor = con.cursor()
        cursor.execute('INSERT INTO records (datetime, img, "dropbox-url") VALUES (?, ?, ?)', [datetime, img, dropbox_url])
        con.commit()
        con.close()
        return cursor.rowcount == 0

    @classmethod
    def get_faces_categories(cls):
        con = cls.get_connection()
        cursor = con.cursor()
        result = cursor.execute('SELECT id, category FROM faces').fetchall()
        con.close()
        result = dict(result)
        return result
