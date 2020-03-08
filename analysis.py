import numpy as np
import pandas as pd
import json
import sys
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from collections import defaultdict, Counter
from networks import DirectedNetwork, UndirectedNetwork


class TweetAnalyzer:
    """This class performs parsing, clustering and analysing of the tweets."""
    def __init__(self):
        self.nlp = spacy.load('en_core_web_sm', disable=['ner'])
        self.nlp.remove_pipe('tagger')
        self.nlp.remove_pipe('parser')
        assert len(self.nlp.pipeline) == 0
        self.tfidf_vectorizer = None
        self.kmeans = None
        self.df = None
        self.retweet_network = DirectedNetwork()
        self.quote_network = DirectedNetwork()
        self.reply_network = DirectedNetwork()
        self.hashtag_network = UndirectedNetwork()
        self.dataset_len = 0

    def tweets_to_data_frame(self, json_tweets):
        """Extracts important information from the json paylaod and loads them to a dataframe."""
        tweets_tmp = list()
        for i, tweet in enumerate(json_tweets):
            if (i + 1) % 1000 == 0:
                print("Successfully parsed %d tweets" % (i + 1))
                sys.stdout.flush()
            self.dataset_len += 1
            if 'created_at' not in tweet:
                # json doesn't contain valid tweet
                continue
            if 'retweeted_status' in tweet:
                retweeter = tweet['user']['screen_name']
                tweet = tweet['retweeted_status']
                orig_tweeter = tweet['user']['screen_name']
                self.retweet_network.update_edge(retweeter, orig_tweeter)

            elif tweet['is_quote_status']:
                quoter = tweet['user']['screen_name']
                tweet = tweet['quoted_status']
                orig_tweeter = tweet['user']['screen_name']
                self.quote_network.update_edge(quoter, orig_tweeter)

            if 'in_reply_to_screen_name' in tweet and tweet['in_reply_to_screen_name'] is not None:
                author = tweet['in_reply_to_screen_name']
                replier = tweet['user']['screen_name']
                self.reply_network.update_edge(replier, author)
            is_truncated = tweet['truncated']

            tweet_text = tweet['extended_tweet']['full_text'] if is_truncated else tweet['text']
            tweet_hashtags_info = tweet['extended_tweet']['entities']['hashtags'] if is_truncated\
                else tweet['entities']['hashtags']
            tweet_hashtags = [x['text'] for x in tweet_hashtags_info]
            num_likes = tweet['favorite_count'] if 'favorite_count' in tweet else 0
            tweets_tmp.append((tweet['id'], tweet['created_at'], tweet['user']['screen_name'], tweet_text,
                               tweet_hashtags, num_likes, tweet['retweet_count']))
            self.hashtag_network.connect_all_vertices(tweet_hashtags)

        labels = ['id', 'created_at', 'user', 'text', 'hashtags', 'likes', 'retweets']
        self.df = pd.DataFrame(data=tweets_tmp, columns=labels)

        self.df.drop_duplicates(subset=['id'], inplace=True)

    def perform_clustering(self, n_clusters):
        """Applies Kmeans algorithm with tfidf features to cluster the tweets stored in the dataframe."""

        # # # VECTOR REPRESENTATION # # #
        tweet_ids, tweet_texts = list(), list()

        for tweet in self.df.itertuples(name='Pandas'):
            tweet_ids.append(tweet.id)
            tweet_texts.append(tweet.text)

        self.tfidf_vectorizer = TfidfVectorizer(sublinear_tf=True,
                                                tokenizer=self.tokenize_normalize, max_features=50000)

        document_term_matrix = self.tfidf_vectorizer.fit_transform(tweet_texts)

        # # # CLUSTERING # # #
        self.kmeans = KMeans(n_clusters=n_clusters, init='k-means++', verbose=10,n_init=5, max_iter=50, n_jobs=-1)
        self.kmeans.fit(document_term_matrix)

    def analyze_clusters(self):
        """Performs analysis of the individual groups as well as the data as a whole."""
        n_clusters = len(self.kmeans.cluster_centers_)

        cluster_contents = defaultdict(list)

        for i, label in enumerate(self.kmeans.labels_):
            cluster_contents[label].append(self.df.iloc[i].text)

        for i in range(n_clusters):
            print("Cluster", i + 1)
            n_samples = 0
            for sample_tweet in cluster_contents[i]:
                if n_samples == 10:
                    break
                print(str(n_samples + 1) + '.', sample_tweet.replace('\n', ' '))
                n_samples += 1
            print()
            if i + 1 == 10:
                print(". . .\n")
                break

        # # # CLUSTER ANALYSIS # # #
        ordered_centroids = self.kmeans.cluster_centers_.argsort()[:, ::-1]
        word_features = self.tfidf_vectorizer.get_feature_names()
        # typical words
        typical_words = [list() for _ in range(n_clusters)]
        for i in range(n_clusters):
            # print("Cluster", i + 1)
            for idx in ordered_centroids[i][:10]:
                # print(word_features[idx])
                typical_words[i].append(word_features[idx])

        # general statistics
        num_posts = [0] * n_clusters
        users = [list() for _ in range(n_clusters)]
        hashtags = [list() for _ in range(n_clusters)]
        like_counts = [list() for _ in range(n_clusters)]
        retweet_counts = [list() for _ in range(n_clusters)]
        for i, label in enumerate(self.kmeans.labels_):
            num_posts[label] += 1
            tweet_info = self.df.iloc[i]
            username = tweet_info.user
            users[label].append(username)
            hashtags[label] += tweet_info.hashtags
            like_counts[label].append(tweet_info.likes)
            retweet_counts[label].append(tweet_info.retweets)

        hashtag_counts = [Counter(hashtags[i]) for i in range(n_clusters)]
        user_counts = [Counter(users[i]) for i in range(n_clusters)]
        num_users = [len(user_counts[i]) for i in range(n_clusters)]
        average_likes = [np.mean(like_counts[i]) for i in range(n_clusters)]
        average_retweets = [np.mean(retweet_counts[i]) for i in range(n_clusters)]

        print("GROUP STATISTICS:\n")
        for i in range(n_clusters):
            print('Cluster %d:' % (i + 1))
            print("Number of tweets:", num_posts[i])
            print("Number of distinct users:", num_users[i])
            print("Typical words in the cluster:", typical_words[i])
            print("Most common users with counts:", user_counts[i].most_common(5))
            print("Most common hashtags with counts:", hashtag_counts[i].most_common(5))
            print("Average number of likes per tweet: %.1f" % average_likes[i])
            print("Average number of retweets per tweet: %.1f" % average_retweets[i])
            print()
            if i + 1 == 10:
                print(". . .\n")
                break
        print("Min tweets: %d, Max tweets: %d" % (np.min(num_posts), np.max(num_posts)))
        print("Min users: %d, Max users: %d" % (np.min(num_users), np.max(num_users)))
        print("Min average likes: %.1f, Max average likes: %.1f" % (np.min(average_likes), np.max(average_likes)))
        print("Min average retweets: %.1f, Max average retweets: %.1f" % (np.min(average_retweets), np.max(average_retweets)))

        words_per_tweet = np.mean([len(text) for text in self.df['text']])

        print("\nOVERALL STATISTICS:\n")

        print("Total number of unique tweets in the dataset: %d" % len(self.df))
        print("Average length of tweet in words: %d" % words_per_tweet)
        print("Number of total retweets: %d (%.1f%% of the initial dataset)" %
              (self.retweet_network.num_connections, 100.0 * self.retweet_network.num_connections / self.dataset_len))
        print("Number of total quotes: %d (%.1f%% of the initial dataset)" %
              (self.quote_network.num_connections, 100.0 * self.quote_network.num_connections / self.dataset_len))
        print("Number of total replies: %d (%.1f%% of the initial dataset)" %
              (self.reply_network.num_connections, 100.0 * self.reply_network.num_connections / self.dataset_len))

        # ties and triads
        print("\nRetweet network:")
        # print("\nRetweet network: ", self.retweet_network.adj_dict)
        num_retweet_ties = self.retweet_network.get_num_edges()
        print("Number of ties: %d" % num_retweet_ties)
        retweet_triads, retweet_closed_triads = self.retweet_network.get_triads()
        print("Triads: ", retweet_triads[:20])
        print("Num of triads: %d , number of closed triads: %d" % (len(retweet_triads), len(retweet_closed_triads)))

        print("\nQuote network:")
        # print("\nQuote network: ", self.quote_network.adj_dict)
        num_quote_ties = self.quote_network.get_num_edges()
        print("Number of ties: %d" % num_quote_ties)
        quote_triads, quote_closed_triads = self.quote_network.get_triads()
        print("Triads: ", quote_triads[:20])
        print("Num of triads: %d , number of closed triads: %d" % (len(quote_triads), len(quote_closed_triads)))

        print("\nReply network:")
        # print("\nReply network: ", self.reply_network.adj_dict)
        num_reply_ties = self.reply_network.get_num_edges()
        print("Number of ties: %d" % num_reply_ties)
        reply_triads, reply_closed_triads = self.reply_network.get_triads()
        print("Triads: ", reply_triads[:20])
        print("Num of triads: %d , number of closed triads: %d" % (len(reply_triads), len(reply_closed_triads)))

        print("\nNumber of ties in total: %d" % (num_quote_ties + num_reply_ties + num_retweet_ties))
        print("Number of triads in total: %d" % (len(reply_triads) + len(retweet_triads) + len(quote_triads)))

        print("\nCo-occurring hash-tags:")
        self.hashtag_network.find_cliques()


    def tokenize_normalize(self, tweets):
        return self.normalize(self.spacy_tokenize(tweets))

    def spacy_tokenize(self, string):
        doc = self.nlp(string)
        tokens = list()
        for token in doc:
            tokens.append(token)

        return tokens

    def normalize(self, tokens):
        normalized = list()

        for token in tokens:
            if token.is_alpha:
                lemma = token.lower_ if token.lemma_ == '-PRON-' else token.lemma_.lower().strip()
                normalized.append(lemma)

        return normalized
