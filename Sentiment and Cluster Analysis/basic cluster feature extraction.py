from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statistics as stat
from scipy.stats import kde

number_of_clusters = 20

documents = []

data_types = {"recommendationid":"string","author":"string", 
             "language" : "string", 
             "timestamp_created":"string", "timestamp_updated":"string", 
             "voted_up":"string", "votes_up":int, "votes_funny":int, 
             "weighted_vote_score":float, "comment_count":int, 
             "steam_purchase":"string", "received_for_free":"string", 
             "written_during_early_access":"string", 
             "timestamp_dev_responded":"string", "developer_response":"string",
             "Text":"string", "target":int, "sentiment confidence":float, "Datetime":"string"}

train = pd.read_csv(r"E:\CHETAN\NITK\ACM\Projects\Sentiment analysis\CP2077_tweets_sentiment.csv", dtype = data_types)
i = 0
for t in train["Text"]:
    if type(t)== pd._libs.missing.NAType:
        train["Text"][i] = " "
    i = i+1


i = 0
'''
for t in train["sentiment confidence"]:
    if type(t)== pd._libs.missing.NAType:
        train.at[i,"sentiment confidence"] = 0.01
        print("Hey")
    train.at[i,"sentiment confidence"] = float(train["sentiment confidence"][i].item())
    i = i+1
'''
print(train["sentiment confidence"][0])

fig = plt.figure(figsize = (8, 5))
predictions = train
predictions = predictions.dropna(subset = ['sentiment confidence'])
prediction = predictions['sentiment confidence']

density = kde.gaussian_kde(prediction)
temp1 = np.linspace(0,1,300)
temp2=density(temp1)

plt.plot(temp1, temp2)
plt.title("Density Plot of the data")
plt.show()

number = len(train["Text"])
n = 0
positive = 0
negative = 0
actual = []
predicted = []
sent = []
for i in range(0,number):
    #print(i)
    if(train["sentiment confidence"][i]<=1) and (train["sentiment confidence"][i]>0.9):
        n += 1
        positive += 1
        sent.append(1)
        predicted.append(1)
        documents.append(train["Text"][i])

print(len(documents))

vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(documents)

'''
# plotting the k vs cluster graph to get the ideal k using elbow method
sum_of_squared_distances = []
K = range(1,150)
for k in K:
    true_k = k
    model = KMeans(n_clusters=true_k, init='k-means++', max_iter=100, n_init=1)
    model = model.fit(X)
    sum_of_squared_distances.append(model.inertia_)

plt.plot(K, sum_of_squared_distances, 'bx-')
plt.xlabel('k')
plt.ylabel('Sum_of_squared_distanes')
plt.title('Elbow method for optimal k')
plt.show()
#'''

true_k = number_of_clusters
model = KMeans(n_clusters=true_k, init='k-means++', max_iter=100, n_init=1)
model.fit(X)


print("Top terms per cluster:")
order_centroids = model.cluster_centers_.argsort()[:, ::-1]
terms = vectorizer.get_feature_names()
for i in range(true_k):
    print("Cluster %d:" % i),
    for ind in order_centroids[i, :10]:
        print(' %s' % terms[ind], end = ', ')
    print('')
    print('')

print("\n")
