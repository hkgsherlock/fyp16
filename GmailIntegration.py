import base64
import mimetypes
import os
from email.mime.audio import MIMEAudio
from email.mime.base import MIMEBase
from email.mime.image import MIMEImage
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from threading import Thread

import httplib2
from googleapiclient import errors, discovery
from oauth2client import clientsecrets
from oauth2client.file import Storage


class GmailIntegration:
    CLIENT_SECRET_FILE = 'client_secret.json'
    SCOPES = ['https://www.googleapis.com/auth/gmail.send',
              'https://www.googleapis.com/auth/gmail.compose']

    STORING_FLOW = None

    TOKEN_FILE_PATH = 'gmail_token.json'

    def __init__(self):
        pass

    @staticmethod
    def create_message(sender, to, subject, message_text):
        """Create a message for an email.
  
        Args:
          sender: Email address of the sender.
          to: Email address of the receiver.
          subject: The subject of the email message.
          message_text: The text of the email message.
  
        Returns:
          An object containing a base64url encoded email object.
        """
        message = MIMEText(message_text)
        message['to'] = to
        message['from'] = sender
        message['subject'] = subject
        return {'raw': base64.urlsafe_b64encode(message.as_string())}

    @staticmethod
    def create_message_with_attachment(
            sender, to, subject, message_text, file):
        """Create a message for an email.
  
        Args:
          sender: Email address of the sender.
          to: Email address of the receiver.
          subject: The subject of the email message.
          message_text: The text of the email message.
          file: The path to the file to be attached.
  
        Returns:
          An object containing a base64url encoded email object.
        """
        message = MIMEMultipart()
        message['to'] = to
        message['from'] = sender
        message['subject'] = subject

        msg = MIMEText(message_text)
        message.attach(msg)

        content_type, encoding = mimetypes.guess_type(file)

        if content_type is None or encoding is not None:
            content_type = 'application/octet-stream'
        main_type, sub_type = content_type.split('/', 1)
        if main_type == 'text':
            fp = open(file, 'rb')
            msg = MIMEText(fp.read(), _subtype=sub_type)
            fp.close()
        elif main_type == 'image':
            fp = open(file, 'rb')
            msg = MIMEImage(fp.read(), _subtype=sub_type)
            fp.close()
        elif main_type == 'audio':
            fp = open(file, 'rb')
            msg = MIMEAudio(fp.read(), _subtype=sub_type)
            fp.close()
        else:
            fp = open(file, 'rb')
            msg = MIMEBase(main_type, sub_type)
            msg.set_payload(fp.read())
            fp.close()
        filename = os.path.basename(file)
        msg.add_header('Content-Disposition', 'attachment', filename=filename)
        message.attach(msg)

        return {'raw': base64.urlsafe_b64encode(message.as_string())}

    @staticmethod
    def create_message_with_image(
            sender, to, subject, message_text, img):
        """Create a message for an email.
  
        Args:
          sender: Email address of the sender.
          to: Email address of the receiver.
          subject: The subject of the email message.
          message_text: The text of the email message.
          file: The path to the file to be attached.
  
        Returns:
          An object containing a base64url encoded email object.
        """
        message = MIMEMultipart()
        message['to'] = to
        message['from'] = sender
        message['subject'] = subject

        msg = MIMEText(message_text)
        message.attach(msg)

        msg = MIMEImage(img, _subtype='jpeg')
        msg.add_header('Content-Disposition', 'attachment', filename='image.jpg')
        message.attach(msg)

        return {'raw': base64.urlsafe_b64encode(message.as_string())}

    @staticmethod
    def send_message(service, user_id, message):
        """Send an email message.
  
        Args:
          service: Authorized Gmail API service instance.
          user_id: User's email address. The special value "me"
          can be used to indicate the authenticated user.
          message: Message to be sent.
  
        Returns:
          Sent Message.
        """
        try:
            message = (service.users().messages().send(userId=user_id, body=message)
                       .execute())
            print 'Message Id: %s' % message['id']
            return message
        except errors.HttpError, error:
            print 'An error occurred: %s' % error

    @staticmethod
    def get_service():
        store = Storage(GmailIntegration.TOKEN_FILE_PATH)
        credentials = store.get()
        if not credentials or credentials.invalid:
            raise clientsecrets.InvalidClientSecretsError
        http = credentials.authorize(httplib2.Http())
        service = discovery.build('gmail', 'v1', http=http)
        return service

    @staticmethod
    def send_email_self(title, context, image=None):
        service = GmailIntegration.get_service()
        user_profile = service.users().getProfile(userId='me').execute()
        user_email = user_profile['emailAddress']
        if image is not None:
            message = GmailIntegration.create_message_with_image(user_email, user_email, title, context, image)
        else:
            message = GmailIntegration.create_message(user_email, user_email, title, context)
        GmailIntegration.send_message(service=service, user_id='me', message=message)

    @staticmethod
    def send_email_self_cv2Mat(title, context, cv2Mat):
        import cv2
        jpg = cv2.imencode('.jpg', cv2Mat)
        GmailIntegration.send_email_self(title, context, jpg)

    @staticmethod
    def notify_confirmed_nowait(name):
        title = "PiSmartCam seees %s" % name
        message = "Hey, the PiSmartCam sees %s! Just to notify you." % name
        t = Thread(target=GmailIntegration.send_email_self, args=[title, message])
        t.daemon = False
        t.start()

    @staticmethod
    def notify_who_nowait(cv2Mat=None):
        title = "guess PiSmartCam sees who?"
        message = "Hey, the PiSmartCam sees someone who are unrecognised! "
        # GmailIntegration.send_email_self_cv2Mat(title, message, cv2Mat)
        if cv2Mat is None:
            t = Thread(target=GmailIntegration.send_email_self, args=[title, message])
        else:
            t = Thread(target=GmailIntegration.send_email_self_cv2Mat, args=[title, message, cv2Mat])
        t.daemon = False
        t.start()

    @staticmethod
    def notify_deny_nowait(cv2Mat=None):
        title = "Someone you denied has trespassed your place"
        message = "Hey, the PiSmartCam sees someone who are denied! "
        # GmailIntegration.send_email_self_cv2Mat(title, message, cv2Mat)
        if cv2Mat is None:
            t = Thread(target=GmailIntegration.send_email_self, args=[title, message])
        else:
            t = Thread(target=GmailIntegration.send_email_self_cv2Mat, args=[title, message, cv2Mat])
        t.daemon = False
        t.start()