class GmailIntegration:
    CLIENT_SECRET_FILE = 'client_secret.json'
    SCOPES = 'https://www.googleapis.com/auth/gmail.send'

    STORING_FLOW = None

    TOKEN_FILE_PATH = 'gmail_token.json'

    def __init__(self):
        pass