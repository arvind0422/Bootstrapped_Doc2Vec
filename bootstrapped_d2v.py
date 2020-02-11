#2#

import json
import pandas as pd
import numpy as np
import re
import sys
import pickle

import keras
from keras.layers import Embedding, Dense,Flatten, Softmax, Input, Dropout
from keras.models import Model
import keras.backend as K

class BDoc2Vec:

    def __init__(self,docs_as_list,max_length,glove_path,hyperparameters=[200,100]):
        self.documents = docs_as_list
        self.max_len = max_length
        self.docvec_size, self.wordvec_size = hyperparameters
        self.num_docs = len(self.documents)

        self.process_vocabulary(glove_path)
        self.get_model()

    # got it online...
    def loadGloveModel(self,gloveFile):
        print("Loading Glove Model")
        f = open(gloveFile,'r')
        model = {}
        for line in f:
            splitLine = line.split()
            word = splitLine[0]
            embedding = np.array([float(val) for val in splitLine[1:]])
            model[word] = embedding
        print("Done.",len(model)," words loaded!")
        return model


    def process_vocabulary(self, glove_path):
        vocabulary = set()
        for ct in self.documents:
            vocabulary = vocabulary.union(set(ct.split()))

        glove_model = self.loadGloveModel(glove_path)

        for word in vocabulary:
            try:    
                blah = glove_model[word]
            except:
                glove_model[word] = np.random.rand(self.wordvec_size,)

        glove_sorted = sorted(glove_model.items(),key=lambda x: x[0])

        t_vocab_to_index = [yy[0] for yy in glove_sorted]
        self.vocab_to_index = dict(zip(t_vocab_to_index,range(len(t_vocab_to_index))))

        self.init_embeddings = np.array([yy[1] for yy in glove_sorted])

    def get_model(self):
        # use hyperparameters to create better models
        # called after processing the vocabulary only.
    
        vocab_size = self.init_embeddings.shape[0]
        input1 = Input(shape=(self.max_len,))
        # vocab_size, init_embeddings, wordvec_size, max_len, docvec_size,num_docs
        embed = Embedding(input_dim=vocab_size, 
                            output_dim=self.wordvec_size, 
                            weights=[self.init_embeddings],
                            trainable=True,
                            input_length=self.max_len)(input1)
        flat = Flatten()(embed)
        d1 = Dense(self.wordvec_size, activation="relu")(flat)
        d2 = Dense(self.docvec_size, activation="relu")(d1)
        d3 = Dense(self.num_docs, activation="relu")(d2)
        output1 = Softmax()(d3)
        
        self.model = Model(input1,output1)
        self.model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
        self.model.summary()
        
    def sequential_permute(self,lst):
        lst = lst.split()
        solution_sequence = []
        for i in range(len(lst)-self.max_len+1):
            solution_sequence.append(" ".join(lst[i:i+self.max_len]))
        return solution_sequence
        

    def forward_permute(self,lst):
        lst = np.array(lst.split())
        solution_sequence = []
        for i in range((len(lst)-self.max_len+1)*3):
            indices = np.sort(np.random.choice(len(lst), self.max_len, replace=False))
            solution_sequence.append(" ".join(list(lst[indices])))
        return solution_sequence

    def random_permute(self,lst):
        lst = np.array(lst.split())
        solution_sequence = []
        for i in range(len(lst)-self.max_len+1):
            indices = np.random.choice(len(lst), self.max_len, replace=False)
            solution_sequence.append(" ".join(list(lst[indices])))
        return solution_sequence

    def generate_data(self):
        labels = []
        train = []
        for idx,doc in enumerate(self.documents):
            
            temp_train = self.sequential_permute(doc) + self.forward_permute(doc) + self.random_permute(doc)
            train += temp_train
            labels += [idx]*len(temp_train)
        
        train_final = np.zeros((len(train),self.max_len))
        for id1,tt in enumerate(train):
            for id2,jj in enumerate(tt.split()):
                train_final[id1,id2] = self.vocab_to_index[jj]
        
        return train_final,np.asarray(labels)


    def fit(self,sims=10,epochs=5):
        # TODO: Early stopping
        for sim in range(sims):
            print("Simulation: {0}".format(sim))
            X_train,y_train = self.generate_data()
            for l_ep in range(epochs):
                history= self.model.fit(X_train,y_train,shuffle=True,epochs=1)
                loss,acc = history.history.values()
                if acc[0] > 0.9:
                    break

    def save(self,fname):
        ff = open(fname,"wb")
        # TODO: save the whole object. pickle would do.
        pickle.dump(self,ff)

    def predict(self,test):
        test_final = np.zeros((len(test),self.max_len))
        for id1,tt in enumerate(test):
            for id2,jj in enumerate(tt.split()):
                test_final[id1,id2] = self.vocab_to_index[jj]
        predictions = self.model.predict(test_final)
        return predictions


def process(in_string):
    # Handle missing data
    if not in_string:
        return "None"
    # Remove non-alpha-numerics, convert to lowercase
    out_string = re.sub('[^-A-Za-z]+', ' ', in_string).lower().split()
    out_words = []
    for oo in out_string:
        if len(oo) <= 2:
            continue
        else:
            out_words.append(oo)
    out_string = " ".join(out_words)
    return out_string

def get_data(path):
    with open (path, "r") as myfile:
        data=myfile.readlines()
    data = json.loads(data[0])["fsb_data"]
    data = pd.DataFrame(data)
    data = data[["file_name","title","symptoms","description"]]
    data["title"] = data["title"].apply(process)
    data["symptoms"] = data["symptoms"].apply(process)
    data["description"] = data["description"].apply(process)
    common_texts=[]
    all_sents = data["symptoms"] # + [" "] + data["title"] + [" "] + data["description"]
    for x in all_sents:
        common_texts.append(x)
    return common_texts

    
if __name__ == "__main__":

    fsb_file = "fsb.txt"
    common_texts = get_data(fsb_file)
    common_texts = common_texts[0:10]

    mxlen = 4
    embeddings_path = "/Users/arvind/Documents/My Documents/Ascendo/NLP/glove.6B/glove.6B.100d.txt"

    model = BDoc2Vec(common_texts, mxlen, embeddings_path)
    model.fit()
    model.save("model_{0}.pkl".format(mxlen))
    predictions = model.predict(["tom"])
    answer = np.argmax(predictions)
