import os
import nltk
import random
from nltk import word_tokenize, WordNetLemmatizer
from nltk.corpus import stopwords
from collections import Counter
from nltk import NaiveBayesClassifier, classify

def init_lists(folder):
    a_list = []
    file_list = os.listdir(folder)
    for a_file in file_list:
        f = open(folder + a_file, 'r', encoding="utf_8")
        a_list.append(f.read())
    f.close()
    return a_list

spam = init_lists('E:/projects/personal/spam-filter/dataset/spam/')
ham = init_lists('E:/projects/personal/spam-filter/dataset/ham/')

all_emails = [(email, spam) for email in spam]
all_emails += [(email, ham) for email in ham]

random.shuffle(all_emails)

def preprocess(sentence):
    lemmatizer = WordNetLemmatizer()
    return [lemmatizer.lemmatize(word.lower()) for word in word_tokenize(sentence)]

stoplist = stopwords.words('english')

def get_features(text, setting):
    if setting == 'bow':
        return {word: count for word, count in Counter(preprocess(text)).items() if not word in stoplist}
    else:
        return {word: True for word in preprocess(text) if not word in stoplist}

all_features = [(get_features(email, 'bow'), label) for (email, label) in all_emails]

def train(features, samples_proportion):
    train_size = int(len(features) * samples_proportion)
    train_set, test_set = features[:train_size], features[train_size:]
    print ('Training set size = ' + str(len(train_set)) + ' emails')
    print ('Test set size = ' + str(len(test_set)) + ' emails')
    train_set_tuple = tuple(train_set)
    classifier = NaiveBayesClassifier.train(train_set_tuple)
    return train_set, test_set, classifier

def evaluate(train_set, test_set, classifier):
    print ('Accuracy on the training set = ' + str(classify.accuracy(classifier, train_set)))
    print ('Accuracy of the test set = ' + str(classify.accuracy(classifier, test_set)))

train_set, test_set, classifier = train(all_features, 0.8)
evaluate(train_set, test_set, classifier)
