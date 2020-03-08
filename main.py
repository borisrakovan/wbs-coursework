from streamer import TwitterStreamer
from analysis import TweetAnalyzer
import authentication

tags = ['music', 'hiphop', 'rap', 'dj', 'singer', 'musician', 'musica', 'artist', 'rock', 'dance', 'livemusic', 'guitar',
        'concert', 'song', 'band', 'producer', 'music love', 'new music', 'now playing', 'radio','music artist',
        'music producer','music video','music studio','soundcloud','podcast','spotify',
        'applemusic','tidal','music streaming','itunes']

"""
Start the whole process by running this function.
"""
if __name__ == "__main__":
    print("Starting the program!")

    tweet_analyzer = TweetAnalyzer()

    auth = authentication.authenticate_twitter_app()
    streamer = TwitterStreamer(auth)

    print("Streaming the tweets...")
    tweets_cursor = streamer.stream_tweets(tags, limit=20000, real_time=False)
    tweet_analyzer.tweets_to_data_frame(tweets_cursor)
    print("Dataframe created, going to perform clustering.")

    n_clusters = 50
    tweet_analyzer.perform_clustering(n_clusters)
    print("\nClusters successfully created. Going to analyse data.")

    # # # ANALYZING GROUPS # # #
    tweet_analyzer.analyze_clusters()
    print("\nThe analysis has finished successfully.")
