import json
from math import log
import numpy as np
from os import listdir
from os.path import isdir
from sys import argv

from nbaux import preprocess

# def normalize(prob_neg_dec, prob_neg_tru, prob_pos_dec, prob_pos_tru):
#     denominator = prob_neg_dec + prob_neg_tru + prob_pos_dec + prob_pos_tru
#     return prob_neg_dec/denominator, prob_neg_tru/denominator, prob_pos_dec/denominator, prob_pos_tru/denominator


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
            
            # prob_neg = log(self.model['prior']['negative'])
            # prob_pos = log(self.model['prior']['positive'])
            # prob_dec = log(self.model['prior']['deceptive'])
            # prob_tru = log(self.model['prior']['truthful'])
            prob_neg_dec = log(self.model['prior']['negative deceptive'])
            prob_neg_tru = log(self.model['prior']['negative truthful'])
            prob_pos_dec = log(self.model['prior']['positive deceptive'])
            prob_pos_tru = log(self.model['prior']['positive truthful'])

            for word in example[0].split():
                if word in self.model['posterior']:
                    # prob_neg += log(self.model['posterior'][word]['negative'] / self.model['posterior'][word]['count'])
                    # prob_pos += log(self.model['posterior'][word]['positive'] / self.model['posterior'][word]['count'])
                    # prob_dec += log(self.model['posterior'][word]['deceptive'] / self.model['posterior'][word]['count'])
                    # prob_tru += log(self.model['posterior'][word]['truthful'] / self.model['posterior'][word]['count'])

                    prob_neg_dec += log(self.model['posterior'][word]['negative deceptive'] / self.model['posterior'][word]['count'])
                    prob_neg_tru += log(self.model['posterior'][word]['negative truthful'] / self.model['posterior'][word]['count'])
                    prob_pos_dec += log(self.model['posterior'][word]['positive deceptive'] / self.model['posterior'][word]['count'])
                    prob_pos_tru += log(self.model['posterior'][word]['positive truthful'] / self.model['posterior'][word]['count'])

            # label_a = 'deceptive' if prob_dec > prob_tru else 'truthful'
            # label_b = 'negative' if prob_neg > prob_pos else 'positive'

            # prob_neg_dec, prob_neg_tru, prob_pos_dec, prob_pos_tru = normalize(prob_neg_dec, prob_neg_tru, prob_pos_dec, prob_pos_tru)
            
            if prob_neg_dec == max(prob_neg_dec, prob_neg_tru, prob_pos_dec, prob_pos_tru):
                label = 'deceptive negative'
            elif prob_neg_tru == max(prob_neg_dec, prob_neg_tru, prob_pos_dec, prob_pos_tru):
                label = 'truthful negative'
            elif prob_pos_dec == max(prob_neg_dec, prob_neg_tru, prob_pos_dec, prob_pos_tru):
                label = 'deceptive positive'
            elif prob_pos_tru == max(prob_neg_dec, prob_neg_tru, prob_pos_dec, prob_pos_tru):
                label = 'truthful positive'

            # self.output += [f'{label_a} {label_b} {path}']
            self.output += [f'{label} {path}']


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