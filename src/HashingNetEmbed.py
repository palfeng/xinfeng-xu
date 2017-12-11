from layers import HashEmbedding, ReduceSum
from keras.layers import Input, Dense, Activation, Embedding, BatchNormalization
from keras.models import Model
import hashlib
import nltk
import keras
import numpy as np
from keras.callbacks import EarlyStopping
from sklearn.datasets import fetch_20newsgroups
import time
import networkx as nx
import random
from random import choice

def get_model(embedding, num_classes):
    input_words = Input([None], dtype='int32', name='input_words')
    print("input_words",input_words)
    
    x = embedding(input_words)
    x = ReduceSum()([x, input_words])
    #print("1:", x)
    #raw_input
    x = Dense(50, activation='relu')(x)
    x = Dense(num_classes)(x)
    x = Activation('softmax')(x)
    model = Model(input=input_words, output=x)
    #print("In get model:" )
    #print(input_words)
    #print(x)
    #raw_input
    return model

def word_encoder(w, max_idx):
    # v = hash(w) #
    v = int(hashlib.sha1(w.encode('utf-8')).hexdigest(), 16)
    return (v % (max_idx-1)) + 1


def fetch_sentences(Prob,Nran,Steps): # use these prob to run random walks and return random walks with their labels. return twenty_train.data(sentences)= [[1,2,3...]...]-list and twerty_train.target(labels)[1 2 3 5...]-ndarray
    path = input("Prob classify: Please input the filename of dataset: ")
    f    =   open(path,'r')
    g0=nx.Graph()
    for line in f:
        oneline = [item.strip() for item in line.split(' ')]   #for AdjNoun.txt: ' ', for karate.txt: '\t'
        g0.add_edge(oneline[0],oneline[1])
        g0[oneline[0]][oneline[1]]['weight']=1  # assume all weights are 1 now

    fetch_sentences.data=[]
    fetch_sentences.target=[]
    for nPr in range(0,len(Prob)):
        for i in range(0,Nran):
            ranNode=choice(g0.nodes())
            infected=[ranNode]
            sent=[int(ranNode)]
            for s in range(0,Steps):
                neighbors=g0.neighbors(ranNode)
                for nei in neighbors:#these nodes already been infected, wont be infected again.
                    if nei in infected:
                        neighbors.remove(nei)
                if len(neighbors) == 0:
                    break
                elif random.random() < Prob[nPr]:
                    ranNode=random.choice(neighbors)
                    infected.append(ranNode)
                    sent.append(int(ranNode))
            fetch_sentences.data.append(sent)
            fetch_sentences.target.append(int(nPr))

    #shuffle two lists together
    temp = list(zip(fetch_sentences.data,fetch_sentences.target))
    random.shuffle(temp)
    fetch_sentences.data,fetch_sentences.target = zip(*temp)
    fetch_sentences.target = np.vstack(fetch_sentences.target)#transfer list to np array
    return fetch_sentences

def fetch_sentences2(Prob, Nran,Steps):
    path = input("Graph color classify: Please input the filename of dataset: ")
    f    =   open(path,'r')
    g0=nx.Graph()
    for line in f:
        oneline = [item.strip() for item in line.split(' ')]   #for AdjNoun.txt: ' ', for karate.txt: '\t'
        g0.add_edge(oneline[0],oneline[1])
        g0[oneline[0]][oneline[1]]['weight']=1  # assume all weights are 1 now

    #print("nodes= ",len(g0.nodes()))
    #print("edges=",len(g0.edges()))
    #raw_input

    #path = input("Graph color classify: Please input attributes file of nodes: ")
    f2    =   open(path+"_att",'r')
    categload={}
    for line in f2:
        oneline = [item.strip() for item in line.split(' ')]
        try:
            categload[oneline[1]].append(oneline[0])
        except KeyError:
            categload[oneline[1]]=[]
            categload[oneline[1]].append(oneline[0])

    #check each category
    #print(categload)
    #raw_input

    fetch_sentences.data=[]
    fetch_sentences.target=[]
    for onecateg in categload.keys():
        gtemp=g0.subgraph(categload[onecateg])
        for i in range(0,Nran):
            ranNode=choice(gtemp.nodes())
            infected=[ranNode]
            sent=[int(ranNode)]
            for s in range(0,Steps):
                neighbors=gtemp.neighbors(ranNode)
                for nei in neighbors:#these nodes already been infected, wont be infected again.
                    if nei in infected:
                        neighbors.remove(nei)
                if len(neighbors) == 0:
                    break
                elif random.random() < Prob:
                    ranNode=random.choice(neighbors)
                    infected.append(ranNode)
                    sent.append(int(ranNode))
            fetch_sentences.data.append(sent)
            fetch_sentences.target.append(int(onecateg))    #careful, the categories should be int-able

    #shuffle two lists together
    temp = list(zip(fetch_sentences.data,fetch_sentences.target))
    random.shuffle(temp)
    fetch_sentences.data,fetch_sentences.target = zip(*temp)
    fetch_sentences.target = np.vstack(fetch_sentences.target)#transfer list to np array
    return fetch_sentences

if __name__ == '__main__':
    startTime=time.time()
    use_hash_embeddings = True
    embedding_size = 20
    num_buckets = 50  # the number of hash result buckets.
    max_words = 5*(10**4)
    max_epochs = 50
    num_hash_functions = 2

    if use_hash_embeddings:
        embedding = HashEmbedding(max_words, num_buckets, embedding_size, num_hash_functions=num_hash_functions)
    else:
        embedding = Embedding(max_words, embedding_size)

    """
    # create dataset1, reading the data as sentences.
    categories = ['alt.atheism', 'soc.religion.christian']#, 'comp.graphics', 'sci.med']
    num_classes = len(categories)
    twenty_train = fetch_20newsgroups(subset='train', categories=categories, shuffle=True, random_state=42)
    data = [nltk.word_tokenize(text)[0:10] for text in twenty_train.data]   #transfer sentences to words array, max size will be 50.
    #print(type(twenty_train.data))
    #print(type(twenty_train.target))


    # data = [nltk.word_tokenize(text) for text in twenty_train.data]. Run on highschool, dutch, airtraffic.txt
    data_encoded = [[word_encoder(w, max_words) for w in text] for text in data]
    print(data_encoded[0:2],type(data_encoded))
    max_len = max([len(d) for d in data])

    # pad data
    data_encoded = [d+[0]*(max_len-len(d)) for d in data_encoded]

    print(data_encoded[0:2],type(data_encoded))
    idx_test = int(len(data_encoded)*0.5)
    data_encoded = np.vstack(data_encoded)                              #pad_docs
    targets = np.asarray(twenty_train.target, 'int32').reshape((-1,1))  #labels
    print(data_encoded[0:2],type(data_encoded))
    """
    """
    # create dataset2, run random walk on graphs as sentences.
    Prob=[0.1,0.5,0.9]
    Nran=200#total random walks on each prob,200,200,500
    Steps=30#total max steps in each random walk,30,30,100
    num_classes = len(Prob)
    # use these prob to run random walks and return random walks with their labels. return twenty_train.data(sentences)= [[1,2,3...]...]-list and twerty_train.target(labels)[1 2 3 5...]-ndarray
    twenty_train = fetch_sentences(Prob,Nran,Steps)

    data_encoded = twenty_train.data
    #print(data_encoded[0:2],type(data_encoded))
    max_len = max([len(d) for d in data_encoded])
    
    # pad data
    data_encoded = [d+[0]*(max_len-len(d)) for d in data_encoded]
    #print(data_encoded[0:2],type(data_encoded))
    idx_test = int(len(data_encoded)*0.5)
    data_encoded = np.vstack(data_encoded)
    #print(data_encoded[0:2],type(data_encoded))                             #pad_docs
    targets = np.asarray(twenty_train.target, 'int32').reshape((-1,1))  #labels
    """
  
    # create dataset3, use groundtruth categories as classifier, use random walk on two subgraph seperately to get sentences. Run on polblogs.txt
    #categ=['0','1']#for polblogs                 #this is only for calculating the num_classes
    categ=['0','1','2','3','4']#for cornell
    Prob=0.8
    Nran=500#total random walks on each prob,200
    Steps=10#total max steps in each random walk,30
    num_classes = len(categ)
    # use these prob to run random walks and return random walks with their labels. return twenty_train.data(sentences)= [[1,2,3...]...]-list and twerty_train.target(labels)[1 2 3 5...]-ndarray
    twenty_train = fetch_sentences2(Prob,Nran,Steps)

    data_encoded = twenty_train.data
    #print(data_encoded[0:2],type(data_encoded))
    max_len = max([len(d) for d in data_encoded])
    
    # pad data
    data_encoded = [d+[0]*(max_len-len(d)) for d in data_encoded]
    #print(data_encoded[0:2],type(data_encoded))
    idx_test = int(len(data_encoded)*0.5)
    data_encoded = np.vstack(data_encoded)
    #print(data_encoded[0:2],type(data_encoded))                             #pad_docs
    targets = np.asarray(twenty_train.target, 'int32').reshape((-1,1))  #labels
 
    
  
    model = get_model(embedding, num_classes)
    metrics = ['accuracy']
    loss = 'sparse_categorical_crossentropy'
    model.compile(optimizer=keras.optimizers.Adam(),loss=loss, metrics=['accuracy'])

    print('Num parameters in model: %i' % model.count_params())
    #set the validation_split argument in model.fit to e.g. 0.1, then the validation data used will be the last 10% of the data
    print(np.shape(data_encoded[0:idx_test,:]),np.shape(targets[0:idx_test]))
    #targets=np.expand_dims(targets,-1)
    model.fit(data_encoded[0:idx_test,:], targets[0:idx_test], validation_split=0.1,nb_epoch=max_epochs)

   # model.fit(data_encoded[0:idx_test,:], targets[0:idx_test], validation_split=0.1,nb_epoch=max_epochs,callbacks=[EarlyStopping(patience=5)])

    test_result = model.test_on_batch(data_encoded[idx_test::, :], targets[idx_test::])
    for i, (name, res) in enumerate(zip(model.metrics_names, test_result)):
        print('%s: %1.4f' % (name, res))
    #output test
    #print("size: ",len(twenty_train.data),len(twenty_train.target),len(data),len(data_encoded))
    #for i in range(0,1):
    #    print("i= ",i)
    #    print("sentence: ", twenty_train.data[i][0:50])
    #    print("target: ", twenty_train.target[i])
    #    #print("data: ", data[i])
    #    print("data encoded: ", data_encoded[i])
    #    print("Embedding: ",model.predict_on_batch(data_encoded[i]))

    endTime=time.time()
    runningTime=endTime-startTime
    print("Running Time= ", runningTime)