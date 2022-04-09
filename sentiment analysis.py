import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statistics as stat
from scipy.stats import kde

data_types = {"recommendationid":"string","author":"string", 
             "language" : "string", 
             "timestamp_created":"string", "timestamp_updated":"string", 
             "voted_up":"string", "votes_up":int, "votes_funny":int, 
             "weighted_vote_score":float, "comment_count":int, 
             "steam_purchase":"string", "received_for_free":"string", 
             "written_during_early_access":"string", 
             "timestamp_dev_responded":"string", "developer_response":"string",
             "Text":"string", "target":int, "sentiment confidence":float, "Datetime":"string"}

train = pd.read_csv(r"E:\CHETAN\NITK\ACM\Projects\Sentiment analysis\train_sentiment_confidence.csv", dtype = data_types)
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
positives = []
negatives = []
sent = []
for i in range(0,number):
    #print(i)
    if(train["sentiment confidence"][i]>0.5):
        n += 1
        positive += 1
        sent.append(1)
        positives.append([train["Datetime"][i], 1])
        
    if (train["sentiment confidence"][i]<=0.5):
        n += 1
        negative += 1
        sent.append(0)
        negatives.append([train["Datetime"][i], 0])

train["sentiment"] = sent
positive = (positive/n)*100
negative = (negative/n)*100
sentiment = [positive,negative]
label = ["Positive", "Negative"]
fig = plt.figure(figsize = (10, 10))
plt.bar(label, sentiment, color ='green', width = 0.4)
plt.xlabel("Sentiment")
plt.ylabel("Percentage of tweets")
plt.title("Overall Sentiment Division for CP2077")
plt.show()

fig = plt.figure(figsize = (12, 7.5))
date = positives[0][0][0:10]
print(date)
dates = []
date_positives = []
number = 0
for p in positives:
    if p[0][0:10] == date:
        number+= 1
    else:
        dates.append(date)
        date_positives.append(number)
        number = 0
        date = p[0][0:10]

#plt.bar(positives[:][0],positives[:][1], color ='maroon')
plt.bar(dates,date_positives,color = "green")
#'''
i = 0
needed_dates = ["2020-11-30", "2020-12-10", "2021-01-10", "2021-02-09"]

for d in dates:
    if d not in needed_dates:
        if(d[8:10]!="00"):
            dates[i] = ""
    i += 1
#'''
x = np.arange(len(dates))
y = list(range(0,25001,5000))
plt.yticks(y)
plt.xticks(x, dates)
plt.xlabel("Date")
plt.ylabel("Number of tweets")
plt.title("Number of positive tweets over time")
plt.show()



fig = plt.figure(figsize = (12, 7.5))
date = negatives[0][0][0:10]
print(date)
dates = []
date_negatives = []
number = 0
for p in negatives:
    if p[0][0:10] == date:
        number+= 1
    else:
        dates.append(date)
        date_negatives.append(number)
        number = 0
        date = p[0][0:10]

#plt.bar(positives[:][0],positives[:][1], color ='maroon')
plt.bar(dates,date_negatives,color = "maroon")
#'''
i = 0
for d in dates:
    if d not in needed_dates:
        if(d[8:10]!="00"):
            dates[i] = ""
    i += 1
#'''
plt.yticks(y)
x = np.arange(len(dates))
plt.xticks(x, dates)
plt.xlabel("Date")
plt.ylabel("Number of tweets")
plt.title("Number of negative tweets over time")
plt.show()


date = negatives[0][0][0:10]
print(date)
dates = []
number = 0
for p in negatives:
    if p[0][0:10] == date:
        number+= 1
    else:
        dates.append(date)
        number = 0
        date = p[0][0:10]
i = 0
date_totals = []
for d in dates:
    date_totals.append(date_negatives[i] + date_positives[i])
    i += 1

fig = plt.figure(figsize = (12, 7.5))
date = positives[0][0][0:10]
print(date)
dates = []
date_positives = []
number = 0
for p in positives:
    if p[0][0:10] == date:
        number+= 1
    else:
        dates.append(date)
        date_positives.append(number)
        number = 0
        date = p[0][0:10]
i = 0
for num in date_positives:
    date_positives[i] = (num/date_totals[i])*100
    i += 1

#plt.bar(positives[:][0],positives[:][1], color ='maroon')
plt.bar(dates,date_positives,color = "green")
#'''
i = 0
#sneeded_dates = ["2020-11-30", "2020-12-10", "2021-01-10", "2021-02-09"]
for d in dates:
    if d not in needed_dates:
        if(d[8:10]!="00"):
            dates[i] = ""
    i += 1
#'''
x = np.arange(len(dates))
y = list(range(0,101,10))
plt.yticks(y)
plt.xticks(x, dates)
plt.xlabel("Date")
plt.ylabel("Percent of of tweets")
plt.title("Percent of positive tweets over time")
plt.show()


fig = plt.figure(figsize = (12, 7.5))
date = negatives[0][0][0:10]
print(date)
dates = []
date_negatives = []
number = 0
for p in negatives:
    if p[0][0:10] == date:
        number+= 1
    else:
        dates.append(date)
        date_negatives.append(number)
        number = 0
        date = p[0][0:10]
i = 0
for num in date_negatives:
    date_negatives[i] = (num/date_totals[i])*100
    i += 1

#plt.bar(positives[:][0],positives[:][1], color ='maroon')
plt.bar(dates,date_negatives,color = "maroon")
#'''
i = 0
for d in dates:
    if d not in needed_dates:
        if(d[8:10]!="00"):
            dates[i] = ""
    i += 1
#'''
plt.yticks(y)
x = np.arange(len(dates))
plt.xticks(x, dates)
plt.xlabel("Date")
plt.ylabel("Percent of tweets")
plt.title("Percent of negative tweets over time")
plt.show()




fig = plt.figure(figsize = (12, 7.5))
date = train["Datetime"][0][0:10]
print(date)
dates = []
date_numbers = []
number = 0
for p in train["Datetime"]:
    if p[0:10] == date:
        number+= 1
    else:
        dates.append(date)
        date_numbers.append(number)
        number = 0
        date = p[0:10]

#plt.bar(positives[:][0],positives[:][1], color ='maroon')
plt.bar(dates,date_numbers,color = "blue")
#'''
i = 0
for d in dates:
    if d not in needed_dates:
        if(d[8:10]!="00"):
            dates[i] = ""
    i += 1
#'''
x = np.arange(len(dates))
y = list(range(0,40001,5000))
plt.yticks(y)
plt.xticks(x, dates)
plt.xlabel("Date")
plt.ylabel("Number of tweets")
plt.title("Tweet volume over time")
plt.show()

