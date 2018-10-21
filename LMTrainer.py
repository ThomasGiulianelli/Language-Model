""" LMTrainer.py
    Thomas Giulianelli
    CSC 470 NLP - Project 2
    10/25/16

    Updated 10/20/18
"""

import collections
import math
import random
from random import randint

# base class for a Language Model
class LanguageModel:

    #constructor
    def __init__(self, textCorpus):
        self.corpus = textCorpus
        lines = open(self.corpus, 'r').readlines() #create list containing full corpus
        c = 1
        self.trainingSet = []
        self.testSet = []
        self.numLines = len(lines)
        self.threshold80 = int(self.numLines * 0.8)  # used to determine end of training set
        self.trainingTokens = ""
        self.numTrainingTokens = 0
        self.unigramProbabilitiesMLE = {}
        self.bigramProbabilitiesMLE = {}
        self.counter1 = None  #unigrams
        self.counter2 = None  #bigrams
        self.unigramProbabilitiesAdd1 = {}
        self.bigramProbabilitiesAdd1 = {}
        self.testTokensString = ""
        self.V = []

        for line in lines:
            c += 1
            line = line.strip()
            if not line:
                continue
            if c < self.threshold80:
                self.trainingSet.append(line) #fill training set
            if c >= self.threshold80:
                self.testSet.append(line) #fill test set
            if c >= self.numLines:
                break

	#prints the name of the corpus
    def printCorpus(self):
        print self.corpus.replace('.txt',':')

    #trains the LM using Maximum Likelihood Estimation
    def trainMLE(self):
        #begin unigram MLE calculations
        self.trainingTokens = " ".join(self.trainingSet) #joins each element of trainingSet (entire lines) into a single string
        self.trainingTokens = self.trainingTokens.split() #create list with individual Tokens as elements
        self.numTrainingTokens = len(self.trainingTokens) #N in the MLE formula

        self.counter1 = collections.Counter(self.trainingTokens) #sets up counter for the trainingTokens list
        unigramTypes = self.counter1.keys() #list of unique words
        unigramFrequencies = self.counter1.values() #list of frequencies corresponding to the list of types
        mostProbableUnigrams = self.counter1.most_common(10) #list of most probable words
        print "Most probable unigrams: %s" % mostProbableUnigrams

        unigramMLE = open(self.corpus.replace('.txt','-') + "MLE-Unigram-Probabilities.txt", "w")

        for word, frequency in self.counter1.iteritems():
            unigramMLE.write(word + " " + str(frequency / float(self.numTrainingTokens)) + "\n") #write probs to file
            self.unigramProbabilitiesMLE[word] = frequency / float(self.numTrainingTokens) #append probs to dictionary

        unigramMLE.close()

        #begin bigram MLE calculations
        bigrams = []
        for i in range(len(self.trainingTokens) - 1):
            bigrams.append(self.trainingTokens[i] + " " + (self.trainingTokens[i + 1])) #fill bigrams list with every bigram in the trainingSet

        self.counter2 = collections.Counter(bigrams) #sets up counter for bigrams list
        bigramTypes = self.counter2.keys() #list of unique bigrams
        bigramFrequencies = self.counter2.values() #list of frequencies corresponding to the list of types
        mostProbableBigrams = self.counter2.most_common(10) #list of most probable bigrams
        print "Most probable bigrams: %s" % mostProbableBigrams

        bigramMLE = open(self.corpus.replace('.txt','-') + "MLE-Bigram-Probabilities.txt", "w")

        for bigram, frequency in self.counter2.iteritems():
            temp = bigram.split()
            firstWordFrequency = self.counter1[temp[0]]
            bigramMLE.write(bigram + " " + str(frequency / float(firstWordFrequency)) + "\n")  #write probs to file
            self.bigramProbabilitiesMLE[bigram] = frequency / float(firstWordFrequency) #append probs to dictionary


    # trains the LM using Add-1 Smoothing (Laplace Smoothing)
    def trainAdd1(self):
        #begin unigram Add1 calculations
        self.testTokensString = " ".join(self.testSet)  # joins each element of testSet (entire lines) into a single string
        testTokens = self.testTokensString.split()  # create list with individual Tokens as elements
        numTestTokens = len(testTokens)

        for token in testTokens:
            if token in self.counter1:
                continue
            self.V.append(token) #create list of tokens that are in the test set but not the training set

        numV = len(self.V)

        unigramAdd1 = open(self.corpus.replace('.txt','-') + "Add1-Unigram-Probabilities.txt", "w")

        for word, frequency in self.counter1.iteritems():
            unigramAdd1.write(word + " " + str((frequency + 1) / float(self.numTrainingTokens + numV)) + "\n") #write probs to file
            self.unigramProbabilitiesAdd1[word] = (frequency + 1) / float(self.numTrainingTokens + numV) #append probs to dictionary

        #begin bigram Add1 calculations
        bigramAdd1 = open(self.corpus.replace('.txt','-') + "Add1-Bigram-Probabilities.txt", "w")

        for bigram, frequency in self.counter2.iteritems():
            temp = bigram.split()
            firstWordFrequency = self.counter1[temp[0]]
            bigramAdd1.write(bigram + " " + str((frequency + 1) / float(firstWordFrequency + numV)) + "\n")  # write probs to file
            self.bigramProbabilitiesAdd1[bigram] = (frequency + 1) / float(firstWordFrequency + numV)  # append probs to dictionary

    # computes the perplexity of the MLE LM (an intrinsic evaluation)
    def computePerplexityMLE(self, testSet):
        testSet = testSet.split()
        perplexity = 1
        N = 0
        default = 0.0000000000000001
        for word in testSet:
            N += 1
            perplexity = perplexity * (1 / self.unigramProbabilitiesMLE.get(word, default))
        perplexity = pow(perplexity, 1 / float(N))
        return perplexity

    #computes the perplexity of the Add1 LM (an intrinsic evaluation)
    def computePerplexityAdd1(self, testSet):
        testSet = testSet.split()
        perplexity = 1
        N = 0
        default = 0.0000000000000001
        for word in testSet:
            N += 1
            perplexity = perplexity * (1 / self.unigramProbabilitiesAdd1.get(word,default))
        perplexity = pow(perplexity, 1 / float(N))
        return perplexity

    #finds and chooses random bigrams (MLE) based on previous word in sentence and returns updated sentence
    def findBigramsMLE(self, firstWord, sentence, iteration):
        mostProbable = 1
        mostProbableBigram = None
        possibleBigram = {}
        iteration += 1
        if iteration > 11:
            return sentence, iteration
        for bigram, probability in self.bigramProbabilitiesMLE.iteritems():
            temp = bigram.split()
            if temp[0] == firstWord:
                possibleBigram[bigram] = 1 #fill subset with bigrams whose first word match the preceding word in the sentence
        mostProbableBigram = random.choice([k for k in possibleBigram for dummy in range(possibleBigram[k])]) #randomly choose a bigram
        if mostProbableBigram:
            temp = mostProbableBigram.split()
            sentence = sentence + " " + temp[1]
            sentence, iteration = self.findBigramsMLE(temp[1], sentence, iteration) # warning: this is recursive
        return sentence, iteration

    # Generates 10 sentences using the MLE LM
    def generateSentencesMLE(self):
        for i in range(10):
            iteration = 1
            sentence = ""
            
            #Weighted Select to choose first word of the sentence
            randomWeight = random.randint(1,self.numTrainingTokens)
            for word,frequency in self.counter1.iteritems():
                randomWeight = randomWeight - frequency
                if  randomWeight <= 0:
                    firstWord = word
                    break

            sentence, iteration = self.findBigramsMLE(firstWord, sentence, iteration)
            print "%s" % sentence + "\n"

    #finds and chooses random bigrams (Add1) based on previous word in sentence and returns updated sentence
    def findBigramsAdd1(self, firstWord, sentence, iteration):
        mostProbable = 1
        mostProbableBigram = None
        possibleBigram = {}
        iteration += 1
        if iteration > 11:
            return sentence, iteration
        for bigram, probability in self.bigramProbabilitiesAdd1.iteritems():
            temp = bigram.split()
            if temp[0] == firstWord:
                possibleBigram[bigram] = 1 #fill subset with bigrams whose first word match the preceding word in the sentence
        mostProbableBigram = random.choice([k for k in possibleBigram for dummy in range(possibleBigram[k])]) #randomly choose a bigram
        if mostProbableBigram:
            temp = mostProbableBigram.split()
            sentence = sentence + " " + temp[1]
            sentence, iteration = self.findBigramsAdd1(temp[1], sentence, iteration) # warning: this is recursive
        return sentence, iteration

    # Generates 10 sentences using the Add1 LM
    def generateSentencesAdd1(self):
        for i in range(10):
            iteration = 1
            sentence = ""

            #Weighted Select to choose first word of the sentence
            randomWeight = random.randint(1,self.numTrainingTokens)
            for word,frequency in self.counter1.iteritems():
                randomWeight = randomWeight - frequency
                if  randomWeight <= 0:
                    firstWord = word
                    break

            sentence, iteration = self.findBigramsAdd1(firstWord, sentence, iteration)
            print sentence + "\n"

def main():
    #create a Language Model based on the Brown corpus
    LM1 = LanguageModel("BrownCorpus.txt")
    LM1.printCorpus()
    LM1.trainMLE()
    LM1.trainAdd1()
    testSet = "The September-October term jury had been charged by Fulton Superior Court" #used for calculating perplexity
    perplexityMLE = LM1.computePerplexityMLE(testSet)
    perplexityAdd1 = LM1.computePerplexityAdd1(testSet)
    print "perplexityMLE = %s" % perplexityMLE
    print "perplexityAdd1 = %s" % perplexityAdd1
    print "Generated Sentences:"
    print "MLE:"
    LM1.generateSentencesMLE()
    print "Add1:"
    LM1.generateSentencesAdd1()
    print "\n"

    #create a Language Model based on the SBC corpus
    LM2 = LanguageModel("sbc.txt")
    LM2.printCorpus()
    LM2.trainMLE()
    LM2.trainAdd1()
    perplexityMLE = LM2.computePerplexityMLE(testSet)
    perplexityAdd1 = LM2.computePerplexityAdd1(testSet)
    print "perplexityMLE = %s" % perplexityMLE
    print "perplexityAdd1 = %s" % perplexityAdd1
    print "Generated Sentences:"
    print "MLE:"
    LM2.generateSentencesMLE()
    print "Add1:"
    LM2.generateSentencesAdd1()

# Execution starts here
if __name__ == '__main__':
    main()
