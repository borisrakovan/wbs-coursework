from pymongo import MongoClient


class DatabaseClient():
    """A class that encapsulated the MongoDB API for more convenient use throughout the application."""

    def __init__(self):
        self.client = MongoClient(
            "mongodb+srv://boris:MyAtZbIYdP6yT23l@twitterdata-gkyjr.mongodb.net/test?retryWrites=true&w=majority&socketTimeoutMS=360000&connectTimeoutMS=360000")
        # db = self.client.test
        # self.client = MongoClient("mongodb://boris:MyAtZbIYdP6yT23l@twitterdata-shard-00-00-gkyjr.mongodb.net:27017,twitterdata-shard-00-01-gkyjr.mongodb.net:27017,twitterdata-shard-00-02-gkyjr.mongodb.net:27017/test?ssl=true&replicaSet=TwitterData-shard-0&authSource=admin&retryWrites=true&w=majority")
        self.database = self.client["twitter_data"]
        self.tweet_collection = self.database["tweets"]
        self.ids = list()

    def push_tweet(self, tweet):
        id_ = self.tweet_collection.insert_one(tweet).inserted_id
        self.ids.append(id_)

    def push_tweets(self, tweets):
        sth = self.tweet_collection.insert_many(tweets)
        ids = sth.inserted_ids
        self.ids += ids

    def get_all_tweets(self, columns=()):
        column_dict = {k: 1 for k in columns}
        column_dict['_id'] = 0
        return self.tweet_collection.find({}, column_dict)

    def query_tweets(self, query, columns=()):
        column_dict = {k: 1 for k in columns}
        column_dict['_id'] = 0
        query = self.tweet_collection.find(query, column_dict)
        return query

    def delete_tweets(self, query):
        return self.tweet_collection.delete_many(query)

    def delete_all(self):
        return self.delete_tweets({})

