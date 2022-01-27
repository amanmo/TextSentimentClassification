#Importing Required Libraries
import json
from math import log
import numpy as np
from os import listdir
from os.path import isdir
from sys import argv

#Importing Custom Library
from nbaux import preprocess

#Function to normalize 4 values
def normalize(a, b, c, d):
    return a / sum([a, b, c, d]), b / sum([a, b, c, d]), c / sum([a, b, c, d]), d / sum([a, b, c, d])
    
class NaiveBayesClassifier:
    '''
    Class that loads a trained model from nbmodel.txt and classifies test data
    '''

    def __init__(self, root):
        'Function to load the trained Naive Bayes model and test data'

        self.model = json.load(open('nbmodel.txt'))
        
        temp = []
        for _ in range(3):
            for level in listdir(root):
                if isdir(root + '/' + level):
                    root = root + '/' + level
        
        for file in listdir(root):
            if file.endswith('.txt'):
                with open(root + '/' + file) as f:
                    temp += [[f.read().strip().lower(), root + '/' + file]]

        self.testData = np.array(temp)
        self.testData = preprocess(self.testData)

    def classify(self):
        'Function to classify the test data'

        self.output = []

        for example in self.testData:

            path = example[1]
            prob_neg_dec = log(self.model['prior']['negative deceptive'])
            prob_neg_tru = log(self.model['prior']['negative truthful'])
            prob_pos_dec = log(self.model['prior']['positive deceptive'])
            prob_pos_tru = log(self.model['prior']['positive truthful'])

            #Calculating Probabilities
            for word in example[0].split():
                if word in self.model['posterior']:
                    prob_neg_dec += log(self.model['posterior'][word]['negative deceptive'] / self.model['posterior'][word]['count'])
                    prob_neg_tru += log(self.model['posterior'][word]['negative truthful'] / self.model['posterior'][word]['count'])
                    prob_pos_dec += log(self.model['posterior'][word]['positive deceptive'] / self.model['posterior'][word]['count'])
                    prob_pos_tru += log(self.model['posterior'][word]['positive truthful'] / self.model['posterior'][word]['count'])

            #Normalizing the Probabilities
            prob_neg_dec, prob_neg_tru, prob_pos_dec, prob_pos_tru = normalize(prob_neg_dec, prob_neg_tru, prob_pos_dec, prob_pos_tru)

            #Classifying based on Probabilities
            label = [None, None]
            if prob_neg_dec == min(prob_neg_dec, prob_neg_tru, prob_pos_dec, prob_pos_tru):
                label = ['deceptive', 'negative']
            elif prob_neg_tru == min(prob_neg_dec, prob_neg_tru, prob_pos_dec, prob_pos_tru):
                label = ['truthful', 'negative']
            elif prob_pos_dec == min(prob_neg_dec, prob_neg_tru, prob_pos_dec, prob_pos_tru):
                label = ['deceptive', 'positive']
            elif prob_pos_tru == min(prob_neg_dec, prob_neg_tru, prob_pos_dec, prob_pos_tru):
                label = ['truthful', 'positive']

            self.output += [f"{' '.join(label)} {path}"]


    def generateOutput(self):
        'Function to write output to nboutput.txt'

        with open('nboutput.txt', 'w') as f:
            f.write('\n'.join(self.output))


if __name__ == '__main__':
    if len(argv) == 1:
        loc = 'Test Data'
        # raise FileNotFoundError
    else:
        loc = argv[1]

    nbc = NaiveBayesClassifier(loc)
    nbc.classify()
    nbc.generateOutput()