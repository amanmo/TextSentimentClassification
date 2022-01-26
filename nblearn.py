import json
import numpy as np
from os import listdir
from os.path import isdir
from sys import argv

from nbaux import preprocess

class NaiveBayesTrainer:
    '''
    Class that does the following:
    1. Parses Input Data and Preprocesses it
    2. Trains a Naive Bayes Classifier (One 4-way classifier)
    3. Saves the learned model to nbmodel.txt
    '''

    @staticmethod
    def parseInput(root, label_1=None, label_2=None):
        'Function to build the numpy array by reading input files'

        temp = []

        for fold in listdir(root):
            if isdir(root + '/' + fold):
                for file in listdir(root + '/' + fold):
                    if file.endswith('.txt'):
                        with open(root + '/' + fold + '/' + file) as f:
                            temp += [[f.read().strip().lower(), label_1, label_2]]

        return np.array(temp)

    def __init__(self, root):
        'Function to traverse through various subfolders and combine the training data'

        negative_deceptive_data = self.parseInput(f'{root}/negative_polarity/deceptive_from_MTurk', 'negative', 'deceptive')
        negative_truthful_data = self.parseInput(f'{root}/negative_polarity/truthful_from_Web', 'negative', 'truthful')
        positive_deceptive_data = self.parseInput(f'{root}/positive_polarity/deceptive_from_MTurk', 'positive', 'deceptive')
        positive_truthful_data = self.parseInput(f'{root}/positive_polarity/truthful_from_TripAdvisor', 'positive', 'truthful')
        
        #Merging numpy arrays
        self.data = np.concatenate((negative_deceptive_data, negative_truthful_data, positive_deceptive_data, positive_truthful_data))

        #Defining Priors
        # self.prior_negative = (len(negative_deceptive_data) + len(negative_truthful_data)) / len(self.data)
        # self.prior_positive = (len(positive_deceptive_data) + len(positive_truthful_data)) / len(self.data)
        # self.prior_truthful = (len(negative_truthful_data) + len(positive_truthful_data)) / len(self.data)
        # self.prior_deceptive = (len(negative_deceptive_data) + len(positive_deceptive_data)) / len(self.data)

        self.prior_neg_dec = len(negative_deceptive_data) / len(self.data)
        self.prior_neg_tru = len(negative_truthful_data) / len(self.data)
        self.prior_pos_dec = len(positive_deceptive_data) / len(self.data)
        self.prior_pos_tru = len(positive_truthful_data) / len(self.data)

        #Preprocessing Data
        self.data = preprocess(self.data)

    def NBTrain(self):
        'Function to train the Naive Bayes Classifier on preprocessed training data'       

        #Find Posterior Probabilities
        #save as number or percentage?
        self.model = {'posterior': {}, 'prior': {
            # 'negative': self.prior_negative,
            # 'positive': self.prior_positive,
            # 'deceptive': self.prior_deceptive,
            # 'truthful': self.prior_truthful
            'negative deceptive': self.prior_neg_dec,
            'negative truthful': self.prior_neg_tru,
            'positive deceptive': self.prior_pos_dec,
            'positive truthful': self.prior_pos_tru
        }}
        for i in range(len(self.data)):
            for word in set(self.data[i][0].split()):
                if word not in self.model['posterior']:
                    # self.model['posterior'][word] = {
                    #     "negative": 1 if self.data[i][1] == "negative" else 0,
                    #     "positive": 1 if self.data[i][1] == "positive" else 0,
                    #     "deceptive": 1 if self.data[i][2] == "deceptive" else 0,
                    #     "truthful": 1 if self.data[i][2] == "truthful" else 0,
                    #     "count": 1
                    # }
                    self.model['posterior'][word] = {
                        "negative deceptive": 1 if self.data[i][1] == "negative" and self.data[i][2] == "deceptive" else 0,
                        "negative truthful": 1 if self.data[i][1] == "negative" and self.data[i][2] == "truthful" else 0,
                        "positive deceptive": 1 if self.data[i][1] == "positive" and self.data[i][2] == "deceptive" else 0,
                        "positive truthful": 1 if self.data[i][1] == "positive" and self.data[i][2] == "truthful" else 0,
                        "count": 1
                    }
                else:
                    # self.model['posterior'][word][self.data[i][1]] += 1
                    # self.model['posterior'][word][self.data[i][2]] += 1
                    self.model['posterior'][word][' '.join([self.data[i][1], self.data[i][2]])] += 1
                    
                    self.model['posterior'][word]['count'] += 1

        #Smoothing (Feature Selection instead?)
        for word in self.model['posterior']:
            for feature in self.model['posterior'][word]:
                if feature != 'count':
                    self.model['posterior'][word][feature] += 1
            # self.model['posterior'][word]['count'] += 2
            self.model['posterior'][word]['count'] += 4

    def saveModel(self):
        'Function to save the trained Naive Bayes Classifier to a human-readable text file'

        json.dump(self.model, open('nbmodel.txt', 'w'))

if __name__ == '__main__':
    if len(argv) == 1:
        loc = 'Sample'
        # raise FileNotFoundError
    else:
        loc = argv[1]
    nbt = NaiveBayesTrainer(loc)
    nbt.NBTrain()
    nbt.saveModel()