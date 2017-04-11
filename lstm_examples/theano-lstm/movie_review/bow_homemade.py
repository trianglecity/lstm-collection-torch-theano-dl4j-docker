
import cPickle as pkl
from collections import Counter
import pandas
import re
import os
import glob
import numpy

dataset_path='./aclImdb/'


def tokenize(sentences):

    tokens = []

    # Lowercase, then replace any non-letter, space, or digit character in the headlines.
    ##new_sentences = [re.sub(r'[^\w\s\d]',' ',h.lower()) for h in sentences]
    
     
    for sentence in sentences:
        ##print ""
	##print "input: ", sentence
	regular_outcome = re.sub(r'[^\w\s\d]',' ', sentence.lower())
	regular_outcome2 = re.sub("\s+", " ", regular_outcome)
	tokens.append(regular_outcome2)
        ##print ""
        ##print "output: ", regular_outcome2    

    # Replace sequences of whitespace with a space character.
    ##new_lines = [re.sub("\s+", " ", h) for h in new_sentences]

    ##for line in new_lines:
    ##	tokens.append(line)

    
    f_stop = open("./stopwords.txt", 'r')
    stopwords = re.split(r'\s+', f_stop.read().lower())
 
    new_tokens = []
    for sentence in tokens:
        print ""
        ##print "stopwords = ", stopwords
	print "input: ", sentence
	new_words = [w for w in re.split(r'\s+', sentence.lower()) if w not in stopwords]
	

    	new_sentence = ""
    	for item in new_words:
		new_sentence = new_sentence + " " + item

        print "output: ", new_sentence
	new_tokens.append(new_sentence)

    return new_tokens



def build_dict(path):

    sentences = []  ### a list for all the reviews

    currdir = os.getcwd()
    
    entry_currdir = currdir

    print "currdit = ", currdir 
    print "path = ", path
    ## path =  ./aclImdb/train

    os.chdir('%s/pos/' % path)
    
    
    for ff in glob.glob("*.txt"):
        print ff
        with open(ff, 'r') as f:
            sentences.append(f.readline().strip())

    print "sentences= ", sentences
    print "currdir = ", os.getcwd()
    ## currdir =  /home/sam/Programming/python/theano/lstm/aclImdb/train/pos
    
    os.chdir(entry_currdir)
    print "currdir = ", os.getcwd()
    ## currdir =  /home/sam/Programming/python/theano/lstm


    os.chdir('%s/neg/' % path)
    for ff in glob.glob("*.txt"):
        print ff
        with open(ff, 'r') as f:
            sentences.append(f.readline().strip())

    ##print "sentences= ", sentences

    ##os.chdir(currdir)
    os.chdir(entry_currdir)
    print "currdir = ", os.getcwd()
    ## currdir =  /home/sam/Programming/python/theano/lstm
    
    ##print "sentences = ", sentences    

    sentences = tokenize(sentences) ## using Python instead of Perl

    print 'Building dictionary..',
    wordcount = dict()
     
    for ss in sentences:
        words = ss.strip().lower().split()
        for w in words:
            if w not in wordcount:
                wordcount[w] = 1
            else:
                wordcount[w] += 1

    counts = wordcount.values()
    keys = wordcount.keys()

    sorted_idx = numpy.argsort(counts)[::-1]

    worddict = dict()
    
    for idx, ss in enumerate(sorted_idx):
        worddict[keys[ss]] = idx+2  # leave 0 and 1 (UNK)

    print numpy.sum(counts), ' total words ', len(keys), ' unique words'

    return worddict


def grab_data(path, dictionary):
    sentences = []
    currdir = os.getcwd()
    os.chdir(path)
    for ff in glob.glob("*.txt"):
        with open(ff, 'r') as f:
            sentences.append(f.readline().strip())

    os.chdir(currdir)
    
    sentences = tokenize(sentences)

    seqs = [None] * len(sentences)
   
    ## for i in range(len(L)):
    ##    item = L[i]

    ## for i, item in enumerate(L):

    
    for idx, ss in enumerate(sentences):
        words = ss.strip().lower().split()
        seqs[idx] = [dictionary[w] if w in dictionary else 1 for w in words]

    return seqs

def main():
    # Get the dataset from http://ai.stanford.edu/~amaas/data/sentiment/
    path = dataset_path
    dictionary = build_dict(os.path.join(path, 'train'))

    ## (stopwords ) Building dictionary.. 2449570  total words  74128  unique words
    ## ("") Building dictionary.. 2515272  total words  120530  unique words
    ## (" ") Building dictionary.. 2674502  total words  74138  unique words

    inv_dictionary = dict((code, word) for word, code in dictionary.items())
    
    train_x_pos = grab_data(path+'train/pos', dictionary)
    
    ###########
    '''
    f = open('imdb_autograd_pos.txt', 'w')
    for i in xrange(len(train_x_pos)):
	bag_of_numbers = train_x_pos[i]
	 
	for number in bag_of_numbers:
		word = inv_dictionary[number]

		f.write(word + " ")
	f.write("\n")
    f.close()
    '''
    
    ###########	
    
    train_x_neg = grab_data(path+'train/neg', dictionary)

    '''
    f = open('imdb_autograd_neg.txt', 'w')
    for i in xrange(len(train_x_neg)):
	bag_of_numbers = train_x_neg[i]
	 
	for number in bag_of_numbers:
		word = inv_dictionary[number]

		f.write(word + " ")
	f.write("\n")
    f.close()
    '''
    ###########
    
    train_x = train_x_pos + train_x_neg
    train_y = [1] * len(train_x_pos) + [0] * len(train_x_neg)

    test_x_pos = grab_data(path+'test/pos', dictionary)
    test_x_neg = grab_data(path+'test/neg', dictionary)
    test_x = test_x_pos + test_x_neg
    test_y = [1] * len(test_x_pos) + [0] * len(test_x_neg)

    f = open('imdb_homemade.pkl', 'wb')
    pkl.dump((train_x, train_y), f, -1)
    pkl.dump((test_x, test_y), f, -1)
    f.close()

    f = open('imdb_homemade.dict.pkl', 'wb')
    pkl.dump(dictionary, f, -1)
    f.close()
    
if __name__ == '__main__':
    main()
