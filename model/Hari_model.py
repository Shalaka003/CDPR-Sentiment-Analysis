import pandas as pd
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import nltk
from nltk.stem import WordNetLemmatizer
wordnet_lemmatizer = WordNetLemmatizer()
from nltk.tokenize.treebank import TreebankWordDetokenizer

split = 0.05
epoch = 6
batch = 32
pad_length = 800
lstm_units = 100
embedding_vector_length = 50

def predict_sentiment(text):
    tw = tokenizer.texts_to_sequences([text])
    tw = pad_sequences(tw,maxlen=pad_length)
    prediction = float(model.predict(tw))
    print("Predicted label: ", prediction)




#train_1= pd.read_csv(r"C:\Users\harik\Documents\CHETAN\NITK\ACM\Projects\Sentiment analysis\Data files\train_clean.csv")

#test = pd.read_csv(r"C:\Users\harik\Documents\CHETAN\NITK\ACM\Projects\Sentiment analysis\Data files\test_clean.csv")
'''

train = pd.concat([train_1, test], axis = 0, join ="outer", ignore_index = True)
train = train.sample(frac = 1).reset_index(drop=True)
print(train['Text'][0])
print(train['Sentiment'][0])


tar = []

for s in train.Sentiment:
    if s=='neg':
        tar.append(0)
    elif s=='pos':
        tar.append(1)

train["target"] = tar

for col in train.columns:
    print(col)
    
print(len(train.index))
'''

'''
tar = []

for s in sim.Sentiment:
    if s=='neg':
        tar.append(0)
    elif s=='pos':
        tar.append(1)

sim["Target"] = tar
#'''


#sentiment_label_train = train["Sentiment"].factorize()
#sentiment_label_test = test.Sentiment.factorize()
#train = pd.DataFrame(columns = ['Text', 'target'])
data_types = {"recommendationid":"string","author":"string", 
             "language" : "string", 
             "timestamp_created":"string", "timestamp_updated":"string", 
             "voted_up":"string", "votes_up":int, "votes_funny":int, 
             "weighted_vote_score":float, "comment_count":int, 
             "steam_purchase":"string", "received_for_free":"string", 
             "written_during_early_access":"string", 
             "timestamp_dev_responded":"string", "developer_response":"string",
             "Text":"string", "target":int}

train = pd.read_csv("training_data.csv", dtype = data_types)
i = 0
for t in train["Text"]:
    if type(t)== pd._libs.missing.NAType:
        train["Text"][i] = ""
    i = i+1
tweet = train.Text.values
m = len(tweet)


from tensorflow.keras.preprocessing.text import Tokenizer
tokenizer = Tokenizer()
tokenizer.fit_on_texts(tweet)
encoded_docs = tokenizer.texts_to_sequences(tweet)
from tensorflow.keras.preprocessing.sequence import pad_sequences
padded_sequence = pad_sequences(encoded_docs, maxlen=pad_length)


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM,Dense, Dropout, SpatialDropout1D
from tensorflow.keras.layers import Embedding

vocab_size = 500000
model = Sequential()
model.add(Embedding(vocab_size, embedding_vector_length, input_length=pad_length))
model.add(SpatialDropout1D(0.25))
model.add(LSTM(lstm_units, dropout=0.5, recurrent_dropout=0.5))
model.add(Dropout(0.2))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy',optimizer='adam', metrics=['accuracy'])
print(model.summary())
import graphviz
from keras.utils.vis_utils import plot_model
plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)


history = model.fit(padded_sequence,train["target"],validation_split=split, epochs=epoch, batch_size=batch, verbose = 1)

print('******************************************************************')
print(f'{split}\t\t\t{epoch}\tbatch {batch}***********************************************************')
print('******************************************************************')

import matplotlib.pyplot as plt
plt.plot(history.history['accuracy'], label='acc')
plt.plot(history.history['val_accuracy'], label='val_acc')
plt.legend()
plt.show()

plt.plot(history.history['loss'], label='loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.legend()
plt.show()

'''
test_sentence1 = "I loved this movie very much"
predict_sentiment(test_sentence1)

test_sentence2 = "This is the worst movie i have ever watched"
predict_sentiment(test_sentence2)
'''
'''
x=0
n =len(train.index)
for i in range(0,n):
    t = train.Text[i]
    if train.Sentiment[i]=="neg":
        x = x+1
        text = " ".join(t.split())
print(x)
word_cloud = WordCloud(max_words = 1000 , width = 1600 , height = 800,
              collocations=False,background_color = 'white').generate(text)

plt.imshow(word_cloud, interpolation='bilinear')
plt.axis("off")
#plt.savefig("myimage.png", dpi=2400)
plt.show()


x=0
n =len(train.index)
for i in range(0,n):
    t = train.Text[i]
    if train.Sentiment[i]=="pos":
        x = x+1
        text = " ".join(t.split())
print(x)
word_cloud = WordCloud(max_words = 1000 , width = 1600 , height = 800,
              collocations=False,background_color = 'white').generate(text)

plt.imshow(word_cloud, interpolation='bilinear')
plt.axis("off")
#plt.savefig("myimage.png", dpi=2400)
plt.show()
'''
