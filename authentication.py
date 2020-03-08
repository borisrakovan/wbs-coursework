from tweepy import OAuthHandler

API_KEY = "R2DpCb61gJ5Zx3ZDuqRyawpBj"
API_SECRET_KEY = "LGjqsgRJnb6CDQL3AolXShdqgb5LSY11Byb4UNWHpb68QWFDiA"
ACCESS_TOKEN = "341625457-ETvI8vzmm6OhXcFjnnx9JmyTSHAAKf3UW8QYMs0l"
ACCESS_TOKEN_SECRET = "uANzKD6ojzPxU0OqvyB5iMpCUXYApt05lxZJM6lsJCypj"


def authenticate_twitter_app():
    auth = OAuthHandler(API_KEY, API_SECRET_KEY)
    auth.set_access_token(ACCESS_TOKEN, ACCESS_TOKEN_SECRET)

    return auth
