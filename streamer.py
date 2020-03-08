from tweepy import API, Cursor, Stream
from tweepy.streaming import StreamListener
from mongo_client import DatabaseClient
import json


class TwitterStreamer:
    """A class that handles streaming of tweets through interaction with the Twitter API"""
    def __init__(self, auth):
        self.auth = auth
        self.db_client = DatabaseClient()
        self.api = API(self.auth)

    def get_user_tweets(self, limit, user):
        tweets = list()

        for tweet in Cursor(self.api.user_timeline, id=user).items(limit):
            tweets.append(tweet)

        return tweets

    def stream_tweets(self, tags, limit=100, real_time=False):
        """This method streams tweets from twitter if real_time=True or tweets previously stored in MongoDB otherwise."""
        if not real_time:
            # try:
            #     with open(filename, 'r') as f:
            #         tweets = [l for l in (line.strip() for line in f) if l]
            # except IOError:
            #     print('Unable to load tweets from file', filename)
            #     raise
            tweets = self.db_client.get_all_tweets().limit(limit)

            return tweets

        self.db_client.delete_all()
        stream_listener = MyStreamListener(self.db_client, limit)
        stream = Stream(self.auth, stream_listener)
        stream.filter(track=tags, languages=['en'])

        return self.db_client.get_all_tweets().limit(limit)


class MyStreamListener(StreamListener):
    """A custom implementation of the tweepy's StreamListener. It stores the received data to MongoDB."""
    def __init__(self, db_client, limit):
        super().__init__()
        self.db_client = db_client
        self.num_tweets = 0
        self.limit = limit

    def on_data(self, raw_data):
        if self.num_tweets % 100 == 0:
            print("%d/%d tweets streamed..." % (self.num_tweets, self.limit))
        # self.tweets.append(raw_data)
        self.num_tweets += 1
        # try:
        #     with open(self.filename, 'w' if self.num_tweets == 1 else 'a') as f:
        #         f.write(raw_data)
        # except BaseException as e:
        #     print("Error while processing data: %s" % (str(e)))
        #     return False
        self.db_client.push_tweet(json.loads(raw_data))
        if self.num_tweets == self.limit:
            return False

    def on_error(self, status_code):
        print("Error while streaming data: %s" % status_code)
        if status_code == 420:
            # we reached the twitter limit
            print("Received 420, should not continue streaming.")
            return False
