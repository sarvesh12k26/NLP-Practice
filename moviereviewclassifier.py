#Movie Review Classification
import random,nltk
from nltk.corpus import movie_reviews

documents=[(list(movie_reviews.words(fileid)),category)
            for category in movie_reviews.categories()
            for fileid in movie_reviews.fileids(category)]

random.shuffle(documents)
print('Number of Documents: {}'.format(len(documents)))
print('First Review: {}'.format(documents[0]))

all_words=[]
for w in movie_reviews.words():
    all_words.append(w.lower())
    
all_words=nltk.FreqDist(all_words)
print('Most Common Words: {}'.format(all_words.most_common(15)))
print('The Word Happy {}'.format(all_words['happy']))  

#Will be using 4000 most common words as features
word_features=list(all_words.keys())[:4000]

#This function determines which of the 4000 words are contained in the review
def find_features(document):
    words=set(document)
    features={}
    for w in word_features:
        features[w]=(w in words)
        
    return features

#this is for one particular document
features=find_features(movie_reviews.words('neg/cv000_29416.txt'))
for key,value in features.items():
    if value==True:
        print(key)

#now for all documents
featuresets=[(find_features(rev) , category) for (rev,category) in documents ]

#Building a model
from sklearn import model_selection
training,testing=model_selection.train_test_split(featuresets,test_size=0.25,random_state=1)

from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.svm import SVC
model=SklearnClassifier(SVC(kernel='linear'))

model.train(training)
accuracy=nltk.classify.accuracy(model,testing)
print('SVC Accuracy: {}'.format(accuracy))
#SVC Accuracy: 0.644



