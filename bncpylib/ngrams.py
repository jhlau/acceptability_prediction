# Ngrams models

import math
from collections import defaultdict
from bncpylib.util import safe_open,tag_separator,FrequencyDistribution,get_tag,get_word,percentile, pointwiseFold, flush_and_close, collect_words
import cPickle
import json
import os
import numpy

# this function is needed just to use the defaultdict container
def defaultdict_int_factory():
    """just a wrapper around a call to defaultdict(int)"""
    return defaultdict(int)

infinity = float('inf')
minus_infinity = - infinity

def log(x):
    """A wrapper around math.log10, so that it returns -inf in the case the values is 0."""
    if x == 0.0:
        return minus_infinity
    else:
        return math.log10(x)

log_base = 10

class LanguageModel(object):
    """A generic language model that actually doesn't do anything"""

    def save(self,index_file):
        """Saves a model to a file"""
        raise NameError('Method save not implemented')

    @classmethod
    def load(klass,attrs,ngrams=None):
        """Loads a model from the JSON attributes parsed from an index file"""
        raise NameError('Method load not implemented')

    def repopulate_from_attrs(self,attrs,ngrams=None):
        """Replaces the content of the current model using attrs"""
        raise NameError('Method repopulate_from_attrs not implemented')
    
    def train(self,train_corpus,order):
        """This method creates the model by reading data from a corpus (a file like object open for reading) and trains a model of the given order"""
        raise NameError('Method train not implemented')

    def logprob(self,sentence):
        """Returns the logprob of a sentence"""
        tokens = self.tokenize_sentence(sentence,self.order)
        return sum(self.tokens_logprob(tokens,self.order))

    def perplexity(self,text):
        """This method calculates the perplexity score for a text. The text is tokenized by this method (and the necessary <s> and </s> tags added)"""
        return log_base ** self.entropy(text)

    def entropy(self,text):
        """This method calculates the entropy score for a text. The text is tokenized by this method (and the necessary <s> and </s> tags added)"""
        tokens = self.tokenize_sentence(text,self.order)
        s = 0.
        delta = self.order - 1
        for i in xrange(delta, len(tokens)):
            ng = tokens[i - delta : i + 1]
            p = self.ngram_prob(ng,self.order)
            s += - log(p)
        return s / float(len(tokens) - (self.order - 1))

    def ngram_prob(self,ngram,order):
        """Returns the probability of a single ngram (probability, not its log). The ngram must be a list of tokens"""
        raise NameError('Method ngram_prob not implemented')
    
    def tokenize_sentence(self,sentence,order):
        """Returns a list of tokens with the correct numbers of initial and end tags (this is meant ot be used with a non-backoff model!!!)"""
        tokens = sentence.split()
        tokens = ['<s>'] * (order-1) + tokens + ['</s>']
        return tokens

    def glue_tokens(self,tokens):
        """The standard way in which we glue together tokens to create keys in our hashmaps"""
        return ' '.join(tokens)

    def tokens_logprob(self,tokens,order):
        """Returns the logprobs assigned by the model to each token for the specified order. The method skips the first order-1 tokens as initial context."""
        delta = self.order - 1
        return [log(self.ngram_prob(tokens[i - delta : i + 1],order)) for i in range(delta,len(tokens))]

    def normalize_logprob_by_sentence_length(self,tokens):
        """Returns the sum of the n-grams logprobs divided by the number of tokens"""
        return sum(self.tokens_logprob(tokens,self.order)) / len(tokens)
    
    def normalize_logprob_by_weighted_sentence_length(self,tokens):
        """Returns the sum of the n-grams logprobs divided by the inverse of the sum of the unigram probabilities"""
        return sum(self.tokens_logprob(tokens,self.order)) / (-1. * sum(self.tokens_logprob(tokens,1)))

    def normalized_min_logprob(self,tokens):
        """Returns the lowest logprob assigned to the n-grams of the sentence divided by its unigram logprob"""
        logprobs = self.tokens_logprob(tokens,self.order)
        unigram_logprobs = self.tokens_logprob(tokens,1)
        return min(map(lambda x,y: - (x / y),logprobs,unigram_logprobs))

    def normalized_max_logprob(self,tokens):
        """Returns the highest logprob assigned to the n-grams of the sentence divided by its unigram logprob"""
        logprobs = self.tokens_logprob(tokens,self.order)
        unigram_logprobs = self.tokens_logprob(tokens,1)
        return max(map(lambda x,y: - (x / y),logprobs,unigram_logprobs))

    def mean_of_n_smallest_ngrams(self,tokens,n=2):
        """Returns the mean value of the n lowest scoring n-grams (normalized by their unigram logprob)"""
        logprobs = self.tokens_logprob(tokens,self.order)
        unigram_logprobs = self.tokens_logprob(tokens,1)
        return sum(sorted(map(lambda x,y: - (x / y),logprobs,unigram_logprobs))[:n]) / n

    def mean_of_first_quartile(self,tokens):
        """Returns the mean of the values are in the first quartile (normalized by the unigram logprobs).
        
        The single n-grams logprob are first normalized, then sorted in ascending order.
        We then compute the first quartile and we return the mean of the normalized values that are less or equal to the quartile.
        """        
        logprobs = self.tokens_logprob(tokens,self.order)
        unigram_logprobs = self.tokens_logprob(tokens,1)
        normalized_logprobs = map(lambda x,y: - (x / y),logprobs,unigram_logprobs)
        quartile = percentile(normalized_logprobs,25.)
        low_values = filter(lambda x : x <= quartile,normalized_logprobs)
        if len(low_values) == 0:
            return normalized_logprobs[0]
        else:
            return sum(low_values) / len(low_values)

    def distance_between_min_max(self,tokens):
        """Returns the distance between the highest and lowest logprobs (normalized by their unigram logprobs)"""
        logprobs = self.tokens_logprob(tokens,self.order)
        unigram_logprobs = self.tokens_logprob(tokens,1)
        minimum = min(map(lambda x,y: - (x / y),logprobs,unigram_logprobs))
        maximum = max(map(lambda x,y: - (x / y),logprobs,unigram_logprobs))
        return (maximum - minimum)

    def syntanctic_log_odds_ratio(self,tokens):
        """This score is taken from Pauls and Klein 2012. It's basically a variation on normalize_logprob_by_weighted_sentence_length"""
        logprobs = self.tokens_logprob(tokens,self.order)
        unigram_logprobs = self.tokens_logprob(tokens,1)
        return (sum(logprobs) - sum(unigram_logprobs)) / len(tokens)

    def mean_of_percentile (self,data, n):
        """Gets the mean of the values lower or equal to the Nth percentile """
        p = percentile(data, n)
        low_values = filter(lambda x : x <= p, data)
        if len(low_values) == 0:
            mean = data[0]
        else:                
            mean = sum(low_values) / len(low_values)        

        return mean
    
    def information_entropy(self, data):
        entropy = 0.0
        for d in data:
            if d != 0.0:
                entropy += -1.0 * d * math.log(d,2)

        return entropy

    # just to avoid having to recalculate the same stuff more than once and to combine them in a single result
    def combined_scores(self,tokens):
        """Returns a list of the scores.
           The advantage of using this function over calling those specific to each score is that all common values are cached which makes the process of computing all the scores much faster."""
        # a bunch of list of of values
        logprobs = self.tokens_logprob(tokens,self.order)
        n = len(logprobs)
        unigram_logprobs = self.tokens_logprob(tokens,1)
        logprobs_normalized_by_unigrams = map(lambda x,y: - (x / y),logprobs,unigram_logprobs)

        norm_min_logprob = min(logprobs_normalized_by_unigrams)
        norm_max_logprob = max(logprobs_normalized_by_unigrams)
        mean_two_smallest = sum(sorted(logprobs_normalized_by_unigrams)[:2]) / 2

        ntrigram_1q = percentile(logprobs_normalized_by_unigrams,25.)
        ntrigram_2q = percentile(logprobs_normalized_by_unigrams,50.)
        ntrigram_mean = sum(logprobs_normalized_by_unigrams)/n
        ntrigram_m1q = self.mean_of_percentile(logprobs_normalized_by_unigrams, 25.)
        ntrigram_m2q = self.mean_of_percentile(logprobs_normalized_by_unigrams, 50.)

        raw_logprob = sum(logprobs)

        norm_logprob = raw_logprob / n

        w_norm_logprob = raw_logprob / (-1. * sum(unigram_logprobs)) # raw_logprob / ((sum(self.tokens_logprob(tokens,1)) * n) / (self.unigram_mean_probability * n))
        w_norm_logprob2 = raw_logprob - sum(unigram_logprobs)

        distance_between_min_and_max = norm_max_logprob - norm_min_logprob

        syntactic_log_odds_ratio = (sum(logprobs) - sum(unigram_logprobs)) / len(tokens)

        min5_ulogprobs = sorted(unigram_logprobs)[:5]
        min5_logprobs = sorted(logprobs)[:5]
        min5_nlogprobs = sorted(logprobs_normalized_by_unigrams)[:5]

        for x in [min5_ulogprobs, min5_logprobs, min5_nlogprobs]:
            if len(x) < 5:
                for i in range(0, 5-len(x)):
                    x.append(0.0)

        return [raw_logprob,sum(unigram_logprobs),norm_logprob,w_norm_logprob,w_norm_logprob2, \
            syntactic_log_odds_ratio] + min5_nlogprobs + \
            [ntrigram_mean,ntrigram_m1q,ntrigram_m2q]



    def combined_scores_csv_header(self):
        """Returns a header for csv output that lists all the scores implemented in this class and in the same order in which they are returned by combined_scores"""
        min5_ulogprobs = "," + ",".join(map(lambda x,y: x + str(y), ["ulogprob-bot-"]*5, range(1,6)))
        min5_logprobs = "," + ",".join(map(lambda x,y: x + str(y), ["logprob-bot-"]*5, range(1,6)))
        min5_nlogprobs = "," + ",".join(map(lambda x,y: x + str(y), ["wlogprob-bot-"]*5, range(1,6)))
        return "logprob,unigram_logprob,mean_logprob,norm_logprob_div,norm_logprob_sub," + \
            "slor" +  min5_nlogprobs + \
            ",wlogprob_mean,wlogprob_m1q,wlogprob_m2q"
                
            
class BasicCountModel(LanguageModel):
    """A basic model that only counts ngrams but doesn't really perform any probability estimation. This is useful just as a base for other models (like one implementing Laplace smoothing)"""

    def __init__(self,train_corpus=None,order=3,cutoff=0,add_unknown_tag=False,timer=None):
        """
        Keyword arguments:

        train_corpus -- if passed to the constructor the model is trained right away (default None)
        order -- the order of the model (3 = trigram, 2 = bigram, etc.) (default 3)
        add_unknown_tag -- if True the model allows for unseen words using an unknown token (default False)
        """
        self.cutoff = cutoff
        self.add_unknown_tag = add_unknown_tag
        self.order = order
        self.timer = timer
        if train_corpus != None:
            self.train(train_corpus,order)
        
    def train(self,train_corpus,order):
        """This method trains the model on train_corpus for ngrams of order order. The method is called automatically by the constructor if a train corpus is provided"""
        self.ngram_maps = defaultdict(defaultdict_int_factory)
        self.total_n_tokens = 0
        self.order = order

        # this is the real training
        for line in train_corpus:
            tokens = self.tokenize_sentence(line,order)
            self.total_n_tokens += len(tokens)
            self.ngram_maps = self.ngrams(tokens,order,self.ngram_maps)
            if not self.timer is None:
                self.timer.advance()    
            
        vocab_inc = 0
        if self.add_unknown_tag:
            vocab_inc = 1

        self.vocab_size = len(self.ngram_maps[1].keys()) + vocab_inc # the size of the vocabulary is the number of unigrams we have seen plus 1, the UNKNOWN tag


        # if we have a cutoff we should remove all the counts that are equal or less the cutoff for the highest order ngrams
        if self.cutoff != 0:
            for t in self.ngram_maps[self.order].keys(): # we need to iterate over keys because we are changing the dict
                if self.ngram_maps[self.order][t] <= self.cutoff:
                    del self.ngram_maps[self.order][t]
        
        
    def ngrams(self,tokens,max_order,ngram_maps):
        """This function counts the ngrams in tokens (a list of things) for each order between 1 and max_order"""
        l = len(tokens)
        for i in xrange(l):
            for d in xrange(1,max_order+1):
                ngram_map = ngram_maps[d]
                j=i+d
                if not j > l:
                    key = self.glue_tokens(tokens[i:j])
                    ngram_map[key] += 1
        return ngram_maps

    def print_tables(self):
        """Print the count table to stdout"""
        for i in xrange(1,self.order + 1):
            print 'Map for {0}-grams'.format(i)
            print self.ngram_maps[i]
            print

    def save(self,index_file):
        """Saves the counts to a file"""
        f = safe_open(index_file,'w')
        model_file_name = os.path.abspath(index_file) + '.model'
        model_file = safe_open(model_file_name, 'w')
        json.dump({'smoothing' : 'none', 'model_file' : model_file_name, 'order': self.order, 'cutoff' : self.cutoff, 'add_unknown_tag' : self.add_unknown_tag},f)
        cPickle.dump(self.ngram_maps,model_file)
        flush_and_close(f)
        flush_and_close(model_file)    
        
    @classmethod
    def load(klass,attrs):
        """Loads a model from the JSON attributes parsed from an index file"""
        f = safe_open(attrs['model_file'],'r')
        maps = cPickle.load(f)
        lm = BasicCountModel()
        lm.order = attrs['order']
        lm.cutoff = attrs['cutoff']
        lm.add_unknown_tag = attrs['add_unknown_tag']
        lm.ngram_maps = maps
        return lm

        
class LaplaceSmoothingModel(BasicCountModel):
    """A language model implementing Laplace (i.e. add 1) smoothing. Not really useful"""    
    def __init__(self,train_corpus=None,order=3,cutoff=0,add_unknown_tag=False,timer=None):
        """
        Keyword arguments:

        train_corpus -- if passed to the constructor the model is trained right away (default None)
        order -- the order of the model (3 = trigram, 2 = bigram, etc.) (default 3)
        add_unknown_tag -- if True the model allows for unseen words using an unknown token (default False)
        """
        super(BasicCountModel,self).__init__(train_corpus,order,cutoff,add_unknown_tag,timer)

    def train(self,train_corpus,order):
        """This method trains the model on train_corpus for ngrams of order order. The method is called automatically by the constructor if a train corpus is provided"""
        super(BasicCountModel,self).train(train_corpus,order)
        self.unigram_denominator = float(self.total_n_tokens + self.vocab_size)

    def ngram_prob(self,ngram,order):
        """Returns the probability of a single ngram (probability, not its log). The ngram must be a list of tokens"""
        if order == 1:
            return (self.ngram_maps[order][self.glue_tokens(ngram)] + 1) / self.unigram_denominator
        else:
            return (self.ngram_maps[order][self.glue_tokens(ngram)] + 1) / float(self.ngram_maps[order - 1][self.glue_tokens(ngram[:-1])] + self.vocab_size)

    def save(self,index_file):
        """Saves a model to a file"""
        f = safe_open(index_file,'w')
        model_file_name = os.path.abspath(index_file) + '.model'
        model_file = safe_open(model_file_name, 'w')
        json.dump({'smoothing' : 'laplace', 'model_file' : model_file_name},f)
        cPickle.dump(self,model_file)
        flush_and_close(f)
        flush_and_close(model_file)
    @classmethod
    def load(klass,attrs):
        """Loads a model from the JSON attributes parsed from an index file"""
        f = safe_open(attrs['model_file'],'r')
        return cPickle.load(f)


class AbstractKneserNeySmoothingModel(LanguageModel):
    """An abstract language model for Kneser Ney interpolated smoothing."""
    def __init__(self,train_corpus=None,order=3,timer=None):
        """
        Keyword arguments:

        train_corpus -- if passed to the constructor the model is trained right away (default None)
        order -- the order of the model (3 = trigram, 2 = bigram, etc.) (default 3)
        """
        self.order = order
        self.timer = timer
        if train_corpus != None:
            self.train(train_corpus,order)

    def glue_tokens(self,tokens,order):
        """In the case of Kneser Ney smoothing we glue tokens differently because we keep different orders in the same dict"""
        if order == 1:
            return u'{0}@{1}'.format(order,tokens)
        else:
            return u'{0}@{1}'.format(order,' '.join(tokens))

    def train(self,train_corpus,order):
        """This method creates the model by reading data from a corpus (a file like object open for reading) and trains a model of the given order"""
        self.order = order
        self.ngram_numerator_map = defaultdict(int)
        self.ngram_denominator_map = defaultdict(int)
        self.unigram_denominator = 0 # for unigrams we don't have contexts
        self.ngram_non_zero_map = defaultdict(int)

        # this is the real training
        for line in train_corpus:
            tokens = self.tokenize_sentence(line,order)
            self.ngrams_interpolated_kneser_ney(tokens,order)
            if not self.timer is None:
                self.timer.advance()

    def ngrams_interpolated_kneser_ney(self,tokens,order):
        """This function counts the n-grams in tokens and also record the lower order non zero counts necessary for interpolated Kneser-Ney smoothing, taken from Goodman 2001 and generalized to arbitrary orders"""
        l = len(tokens)
        for i in xrange(order-1,l): # tokens should have a prefix of order - 1 <s>s
            for d in xrange(order,0,-1):
                if d == 1:
                    self.unigram_denominator += 1
                    self.ngram_numerator_map[self.glue_tokens(tokens[i],d)] += 1
                else:
                    den_key = self.glue_tokens(tokens[i-(d-1) : i],d)
                    num_key = self.glue_tokens(tokens[i-(d-1) : i+1],d)
                    self.ngram_denominator_map[den_key] += 1
                    tmp = self.ngram_numerator_map[num_key] # we store this value to check if it's 0
                    self.ngram_numerator_map[num_key] += 1 # we increment it
                    if tmp == 0: # if this is the first we this ngram
                        self.ngram_non_zero_map[den_key] += 1 # we increment the non zero count and implicitly switch to the lower order
                    else:
                        break # if the ngram has already been seen we don't go down to lower order models

    def pp(self):
        """Pretty print"""
        print 'Order:',self.order
        print 'Unigram denominator:',self.unigram_denominator
        print 'Ngram numerator:'
        for key,val in self.ngram_numerator_map.iteritems():
            print ' ', key,val
        print 'Ngram denominator:'
        for key,val in self.ngram_denominator_map.iteritems():
            print ' ', key,val            
        print 'Non zero:'
        for key,val in self.ngram_non_zero_map.iteritems():
            print ' ', key,val

class KneserNeySmoothingModel(AbstractKneserNeySmoothingModel):
    """The standard implementation of Kneser Ney interpolation"""
    def __init__(self,train_corpus=None,order=3,timer=None):
        """
        Keyword arguments:

        train_corpus -- if passed to the constructor the model is trained right away (default None)
        order -- the order of the model (3 = trigram, 2 = bigram, etc.) (default 3)
        """
        super(KneserNeySmoothingModel, self).__init__(train_corpus,order,timer)


    def raw_ngram_prob(self,ngram,discount,order):
        """The internal implementation of ngram_prob.

           We could do without it but if we want, for some reason, to use a different discount than the general one we can do it with this function.
        """
        # compute the unigram prob
        probability = previous_prob = self.ngram_numerator_map[self.glue_tokens(ngram[-1],1)] / float(self.unigram_denominator)
        
        # now we compute the higher order probs and interpolate
        for d in xrange(2,order+1):
            ngram_den = self.ngram_denominator_map[self.glue_tokens(ngram[-(d):-1],d)]
            if ngram_den != 0:
                ngram_num = self.ngram_numerator_map[self.glue_tokens(ngram[-(d):],d)]
                if ngram_num != 0:
                    current_prob = (ngram_num - discount) / float(ngram_den)
                else:
                    current_prob = 0.0
                current_prob += self.ngram_non_zero_map[self.glue_tokens(ngram[-(d):-1],d)] * discount / ngram_den * previous_prob
                previous_prob = current_prob
                probability = current_prob
            else:
                probability = previous_prob
                break

        return probability

    def ngram_prob(self,ngram,order):
        """Calculates the logprob of a single ngram. ngram must be a list of tokens (the order parameter is there just to avoid having to compute it). Taken from Goodman 2001 and generalized to arbitrary orders"""
        return self.raw_ngram_prob(ngram,self.discount,order)

    def repopulate_from_attrs(self,attrs,ngrams=None):
        """Loads a Kneser-Ney model"""

        def selective_update(m,k,v):
            if k in ngrams:
                m[k] = v

        def unrestricted_update(m,k,v):
            m[k] = v
        
        self.order = attrs['order']
        self.discount = attrs['discount']
        self.unigram_denominator = attrs['unigram_den']
        self.ngram_numerator_map = defaultdict(int)
        self.ngram_denominator_map = defaultdict(int)
        self.ngram_non_zero_map = defaultdict(int)

        if ngrams is None:
            update = unrestricted_update
        else:
            update = selective_update        
        
        flag = True
        db = open(attrs['num_db'],'r')
        while flag:
            try:
                k = cPickle.load(db)
                v = cPickle.load(db)
                update(self.ngram_numerator_map,k,v)
            except EOFError:
                flag = False

        flag = True
        db = open(attrs['den_db'],'r')
        while flag:
            try:
                k = cPickle.load(db)
                v = cPickle.load(db)
                update(self.ngram_denominator_map,k,v)
            except EOFError:
                flag = False

        flag = True
        db = open(attrs['nz_db'],'r')
        while flag:
            try:
                k = cPickle.load(db)
                v = cPickle.load(db)
                update(self.ngram_non_zero_map,k,v)
            except EOFError:
                flag = False

        return self
    
    @classmethod
    def load(klass,attrs,ngrams=None):
        """Loads a Kneser-Ney model"""

        def selective_update(m,k,v):
            if k in ngrams:
                m[k] = v
        def unrestricted_update(m,k,v):
            m[k] = v
        
        self = klass()
        self.order = attrs['order']
        self.discount = attrs['discount']
        self.unigram_denominator = attrs['unigram_den']
        self.ngram_numerator_map = defaultdict(int)
        self.ngram_denominator_map = defaultdict(int)
        self.ngram_non_zero_map = defaultdict(int)

        if ngrams is None:
            update = unrestricted_update
        else:
            update = selective_update        
        
        flag = True
        db = open(attrs['num_db'],'r')
        while flag:
            try:
                k = cPickle.load(db)
                v = cPickle.load(db)
                update(self.ngram_numerator_map,k,v)
            except EOFError:
                flag = False

        flag = True
        db = open(attrs['den_db'],'r')
        while flag:
            try:
                k = cPickle.load(db)
                v = cPickle.load(db)
                update(self.ngram_denominator_map,k,v)
            except EOFError:
                flag = False

        flag = True
        db = open(attrs['nz_db'],'r')
        while flag:
            try:
                k = cPickle.load(db)
                v = cPickle.load(db)
                update(self.ngram_non_zero_map,k,v)
            except EOFError:
                flag = False

        return self

    def save(self,index_filename):
        """Saves a Kneser-Ney model"""
        f = safe_open(index_filename,'wb')
        num_db = os.path.abspath(index_filename) + '.num.db'
        den_db = os.path.abspath(index_filename) + '.den.db'
        nz_db = os.path.abspath(index_filename) + '.nz.db'

        db = open(num_db,'w')
        for k,v in self.ngram_numerator_map.iteritems():
            cPickle.dump(k,db,-1)
            cPickle.dump(v,db,-1)
        db.close()

        db = open(den_db,'w')
        for k,v in self.ngram_denominator_map.iteritems():
            cPickle.dump(k,db,-1)
            cPickle.dump(v,db,-1)
        db.close()

        db = open(nz_db,'w')
        for k,v in self.ngram_non_zero_map.iteritems():
            cPickle.dump(k,db,-1)
            cPickle.dump(v,db,-1)
        db.close()

        json.dump({'smoothing' : 'kneser-ney', 'discount' : self.discount, 'num_db' : num_db, 'den_db' : den_db, 'nz_db' : nz_db, 'order' : self.order, 'unigram_den' : self.unigram_denominator},f)
        flush_and_close(f)

class StingyKneserNeySmoothingModel(KneserNeySmoothingModel):
    def __init__(self,train_corpus=None,test_corpus=None,order=3,timer=None):
        self.order = order
        self.timer = timer
        if train_corpus != None:
            self.train(train_corpus,test_corpus,order)

    def train(self,train_corpus,test_corpus,order):
        """This method creates the model by reading data from a corpus (a file like object open for reading) and trains a model of the given order"""
        self.order = order
        self.ngram_numerator_map = defaultdict(int)
        self.ngram_denominator_map = defaultdict(int)
        self.unigram_denominator = 0 # for unigrams we don't have contexts
        self.ngram_non_zero_map = defaultdict(int)

        if test_corpus is None:
            voc = set()
        else:
            voc = collect_words(test_corpus)
            voc.add('<s>')
            voc.add('</s>')

        # this is the real training
        for line in train_corpus:
            tokens = self.tokenize_sentence(line,order)
            self.ngrams_interpolated_kneser_ney(tokens,order,voc)
            if not self.timer is None:
                self.timer.advance()

    def ngrams_interpolated_kneser_ney(self,tokens,order,voc):
        """This function counts the n-grams in tokens and also record the lower order non zero counts necessary for interpolated Kneser-Ney smoothing, taken from Goodman 2001 and generalized to arbitrary orders"""
        l = len(tokens)
        for i in xrange(order-1,l): # tokens should have a prefix of order - 1 <s>s
            for d in xrange(order,0,-1):
                if tokens[i] in voc:    
                    if d == 1:
                        self.unigram_denominator += 1
                        self.ngram_numerator_map[self.glue_tokens(tokens[i],d)] += 1
                    else:
                        den_key = self.glue_tokens(tokens[i-(d-1) : i],d)
                        num_key = self.glue_tokens(tokens[i-(d-1) : i+1],d)
                        self.ngram_denominator_map[den_key] += 1
                        tmp = self.ngram_numerator_map[num_key] # we store this value to check if it's 0
                        self.ngram_numerator_map[num_key] += 1 # we increment it
                        if tmp == 0: # if this is the first we this ngram
                            self.ngram_non_zero_map[den_key] += 1 # we increment the non zero count and implicitly switch to the lower order
                        else:
                            break # if the ngram has already been seen we don't go down to lower order models

    def raw_ngram_prob(self,ngram,discount,order):
        """The internal implementation of ngram_prob.

           We could do without it but if we want, for some reason, to use a different discount than the general one we can do it with this function.
        """
        # compute the unigram prob
        probability = previous_prob = self.ngram_numerator_map[self.glue_tokens(ngram[-1],1)] / float(self.unigram_denominator)
        
        # now we compute the higher order probs and interpolate
        for d in xrange(2,order+1):
            ngram_den = self.ngram_denominator_map[self.glue_tokens(ngram[-(d):-1],d)]
            if ngram_den != 0:
                ngram_num = self.ngram_numerator_map[self.glue_tokens(ngram[-(d):],d)]
                if ngram_num != 0:
                    current_prob = (ngram_num - discount) / float(ngram_den)
                else:
                    current_prob = 0.0
                current_prob += self.ngram_non_zero_map[self.glue_tokens(ngram[-(d):-1],d)] * discount / ngram_den * previous_prob
                previous_prob = current_prob
                probability = current_prob
            else:
                probability = previous_prob
                break

        if probability == 0.0:
            print ngram
            print self.ngram_numerator_map[self.glue_tokens(ngram[-1],1)]    
            
        return probability
                            
    def save(self,index_filename):
        """Saves a Kneser-Ney model"""
        f = safe_open(index_filename,'wb')
        num_db = os.path.abspath(index_filename) + '.num.db'
        den_db = os.path.abspath(index_filename) + '.den.db'
        nz_db = os.path.abspath(index_filename) + '.nz.db'

        db = open(num_db,'w')
        for k,v in self.ngram_numerator_map.iteritems():
            cPickle.dump(k,db,-1)
            cPickle.dump(v,db,-1)
        db.close()

        db = open(den_db,'w')
        for k,v in self.ngram_denominator_map.iteritems():
            cPickle.dump(k,db,-1)
            cPickle.dump(v,db,-1)
        db.close()

        db = open(nz_db,'w')
        for k,v in self.ngram_non_zero_map.iteritems():
            cPickle.dump(k,db,-1)
            cPickle.dump(v,db,-1)
        db.close()

        json.dump({'smoothing' : 'stingy-kneser-ney', 'discount' : self.discount, 'num_db' : num_db, 'den_db' : den_db, 'nz_db' : nz_db, 'order' : self.order, 'unigram_den' : self.unigram_denominator},f)
        flush_and_close(f)

                            
class TagBasedKneserNeySmoothingModel(KneserNeySmoothingModel):
    """An extension to Kneser-Ney smoothing to train tagged or clustered corpora"""    
    # we override this methods because we need to tag the tags!
    def tokenize_sentence(self,sentence,order):
        """Returns a list of tokens with the correct numbers of initial and end tags (this is meant to be used with a non-backoff model!!!)"""
        tokens = sentence.split()
        tokens = ['<s>' + tag_separator + '<s>'] * (order-1) + tokens + ['</s>' + tag_separator + '</s>']
        return tokens

    def train(self,train_corpus,order):
        """This method creates the model by reading data from a corpus (a file like object open for reading) and trains a model of the given order"""
        self.order = order
        self.ngram_numerator_map = defaultdict(int)
        self.ngram_denominator_map = defaultdict(int)
        self.unigram_denominator = 0 # for unigrams we don't have contexts
        self.ngram_non_zero_map = defaultdict(int)

        self.words_freq_dist = FrequencyDistribution()
        self.tags_freq_dist = FrequencyDistribution()

        # this is the real training
        for line in train_corpus:
            tokens = self.tokenize_sentence(line,order)
            tag_tokens = map(get_tag,tokens)
            word_tokens = map(get_word,tokens)
            self.ngrams_interpolated_kneser_ney(tag_tokens,order)
            self.words_freq_dist.add_counts(word_tokens)
            self.tags_freq_dist.add_counts(tag_tokens)
            if not self.timer is None:
                self.timer.advance()

    def ngram_prob(self,ngram,order):
        """We modify how the probability of an n-gram is computed. It is immaterial whether we interpolate the final probabilities or just the tags probabilities due to the following equivalence (here exemplified only for trigrams)

        P(w_i | w_i-2 w_i-1) = 
        P(w_i | C_i) * P(C_i | C_i-2 C_i-1) =
        P(w_i | C_i) * (P(C_i | C_i-2 C_i-1) + l_1 * P(C_i | C_i-1) + l_2 * P(C_i)) =
        P(w_i | C_i) * P(C_i | C_i-2 C_i-1) + P(w_i | C_i) * l_1 * P(C_i | C_i-1) + P(w_i | C_i) * l_2 * P(C_i) =
        P(w_i | w_i-2 w_i-1) + l_1 * P(w_i | w_i-1) + l_2 * P(w_i) =
        P(w_i | w_i-2 w_i-1)
        """

        tags_ngram = map(get_tag,ngram)
        word = get_word(ngram[-1])
        tag = get_tag(ngram[-1])

        return (float(self.words_freq_dist[word] + 1) / self.tags_freq_dist[tag]) * self.raw_ngram_prob(tags_ngram,self.discount,order)
 
    def save(self,index_filename):
        """Saves the model to a file"""
        f = safe_open(index_filename,'wb')
        num_db = os.path.abspath(index_filename) + '.num.db'
        den_db = os.path.abspath(index_filename) + '.den.db'
        nz_db = os.path.abspath(index_filename) + '.nz.db'
        frequencies_db = os.path.abspath(index_filename) + '.freq.db'

        db = open(frequencies_db,'w')
        words_freq_count = 0
        for k,v in self.words_freq_dist.iteritems():
            cPickle.dump(k,db,-1)
            cPickle.dump(v,db,-1)
            words_freq_count += 1

        tags_freq_count = 0
        for k,v in self.tags_freq_dist.iteritems():
            cPickle.dump(k,db,-1)
            cPickle.dump(v,db,-1)
            tags_freq_count += 1
        db.close()
        

        db = open(num_db,'w')
        for k,v in self.ngram_numerator_map.iteritems():
            cPickle.dump(k,db,-1)
            cPickle.dump(v,db,-1)
        db.close()

        db = open(den_db,'w')
        for k,v in self.ngram_denominator_map.iteritems():
            cPickle.dump(k,db,-1)
            cPickle.dump(v,db,-1)
        db.close()

        db = open(nz_db,'w')
        for k,v in self.ngram_non_zero_map.iteritems():
            cPickle.dump(k,db,-1)
            cPickle.dump(v,db,-1)
        db.close()

        json.dump({'smoothing' : 'tag-kneser-ney', 'discount' : self.discount, 'num_db' : num_db, 'den_db' : den_db, 'nz_db' : nz_db, 'order' : self.order, 'unigram_den' : self.unigram_denominator, 'words_freq_count' : words_freq_count, 'tags_freq_count' : tags_freq_count, 'frequencies_db' : frequencies_db},f)

        flush_and_close(f)


    @classmethod
    def load(klass,attrs):
        """Loads a Kneser-Ney model"""
        self = klass()
        self.order = attrs['order']
        self.discount = attrs['discount']
        self.unigram_denominator = attrs['unigram_den']
        self.ngram_numerator_map = defaultdict(int)
        self.ngram_denominator_map = defaultdict(int)
        self.ngram_non_zero_map = defaultdict(int)

        flag = True
        db = open(attrs['num_db'],'r')
        while flag:
            try:
                k = cPickle.load(db)
                v = cPickle.load(db)
                self.ngram_numerator_map[k] = v
            except EOFError:
                flag = False

        flag = True
        db = open(attrs['den_db'],'r')
        while flag:
            try:
                k = cPickle.load(db)
                v = cPickle.load(db)
                self.ngram_denominator_map[k] = v
            except EOFError:
                flag = False

        flag = True
        db = open(attrs['nz_db'],'r')
        while flag:
            try:
                k = cPickle.load(db)
                v = cPickle.load(db)
                self.ngram_non_zero_map[k] = v
            except EOFError:
                flag = False

        self.words_freq_dist = FrequencyDistribution()
        self.tags_freq_dist = FrequencyDistribution()


        words_freq_count = attrs['words_freq_count']
        tags_freq_count = attrs['tags_freq_count']
        db = open(attrs['frequencies_db'],'r')

        while words_freq_count > 0:
            k = cPickle.load(db)
            v = cPickle.load(db)
            self.words_freq_dist[k] = v
            words_freq_count -= 1

        while tags_freq_count > 0:
            k = cPickle.load(db)
            v = cPickle.load(db)
            self.tags_freq_dist[k] = v
            tags_freq_count -= 1

        return self

class ContextNgramKneserNey(KneserNeySmoothingModel):
    """An untested experiment with smoothing a Markov random field"""
    def __init__(self,train_corpus=None,order=3,timer=None):
        super(ContextNgramKneserNey, self).__init__(train_corpus,order,timer)

    def glue_tokens(self,tokens,order):
        return u'{0}@{1}'.format(order,' '.join(tokens))

    def left_right_ngram(self,tokens,index,order):
        return tokens[((index-order)+1) : (index + order)]

    def ncontext(self,tokens,index,order):
        return tokens[((index-order)+1) : index] + tokens[(index+1) : (index + order)]

    def tokens_logprob(self,tokens,order):
        """Returns the logprobs assigned by the model to each token for the specified order. The method skips the first order-1 tokens as initial context."""
        delta = self.order - 1
        return [log(self.ngram_prob(self.left_right_ngram(tokens,i,order),order)) for i in range(delta,len(tokens)-order)]

    # we need to override this method to get the correct number of sentence end tags
    def tokenize_sentence(self,sentence,order):
        """Returns a list of tokens with the correct numbers of initial and end tags (this is meant ot be used with a non-backoff model!!!)"""
        tokens = sentence.split()
        tokens = ['<s>'] * (order-1) + tokens + ['</s>'] * (order-1)
        return tokens

    # we override this method because it works on single ngrams
    def entropy(self,text):
        """This method calculates the perplexity score for a text. The text is tokenized by this method (and the necessary <s> and </s> tags added)"""
        tokens = self.tokenize_sentence(text,self.order)
        s = 0.
        delta = self.order - 1
        for i in xrange(delta, len(tokens)):
            ng = self.left_right_ngram(tokens,i,self.order)
            p = self.ngram_prob(ng,self.order)
            s += - log(p)
        return s / float(len(tokens) - (self.order - 1))

    def ngrams_interpolated_kneser_ney(self,tokens,order):
            """This function counts the ngram in tokens and also record the lower order non zero counts necessary for interpolated Kneser-Ney smoothing, taken from Goodman 2001 and generalized to arbitrary orders"""
            l = len(tokens)
            for i in xrange(order-1,l-order): # tokens should have a prefix of order - 1 <s>s and a suffix of order - 1 </s>s
                for d in xrange(order,0,-1):
                    if d == 1:
                        self.unigram_denominator += 1
                        self.ngram_numerator_map[self.glue_tokens(tokens[i],d)] += 1
                    else:
                        den_key = self.glue_tokens(self.ncontext(tokens,i,d),d)
                        num_key = self.glue_tokens(self.left_right_ngram(tokens,i,d),d)
                        self.ngram_denominator_map[den_key] += 1
                        tmp = self.ngram_numerator_map[num_key] # we store this value to check if it's 0
                        self.ngram_numerator_map[num_key] += 1 # we increment it
                        if tmp == 0: # if this is the first we this ngram
                            self.ngram_non_zero_map[den_key] += 1 # we increment the non zero count and implicitly switch to the lower order
                        else:
                            break # if the ngram has already been seen we don't go down to lower order models

    def raw_ngram_prob(self,ngram,discount,order):
        """Calculates the logprob of a single ngram. ngram must be a list of tokens (the order parameter is there just to avoid having to compute it). Taken from Goodman 2001 and generalized to arbitrary orders"""
        # compute the unigram prob
        probability = previous_prob = self.ngram_numerator_map[self.glue_tokens(ngram[-1],1)] / float(self.unigram_denominator)
        
        # now we compute the higher order probs and interpolate
        for d in xrange(2,order+1):
            pivot = order - 1
            pivot_val = ngram[pivot]
            left_context = ngram[:pivot]
            right_context = ngram[order:]
            den_key = left_context[-(d-1):] + right_context[:(d-1)]
            num_key = left_context[-(d-1):] + [pivot_val] + right_context[:(d-1)]
            ngram_den = self.ngram_denominator_map[self.glue_tokens(den_key,d)]
            if ngram_den != 0:
                ngram_num = self.ngram_numerator_map[self.glue_tokens(num_key,d)]
                if ngram_num != 0:
                    current_prob = (ngram_num - discount) / float(ngram_den)
                else:
                    current_prob = 0.0
                current_prob += self.ngram_non_zero_map[self.glue_tokens(den_key,d)] * discount / ngram_den * previous_prob
                previous_prob = current_prob
                probability = current_prob
            else:
                probability = previous_prob
                break

        return probability

    def save(self,index_filename):
        f = safe_open(index_filename,'wb')
        num_db = os.path.abspath(index_filename) + '.num.db'
        den_db = os.path.abspath(index_filename) + '.den.db'
        nz_db = os.path.abspath(index_filename) + '.nz.db'

        db = open(num_db,'w')
        for k,v in self.ngram_numerator_map.iteritems():
            cPickle.dump(k,db,-1)
            cPickle.dump(v,db,-1)
        db.close()

        db = open(den_db,'w')
        for k,v in self.ngram_denominator_map.iteritems():
            cPickle.dump(k,db,-1)
            cPickle.dump(v,db,-1)
        db.close()

        db = open(nz_db,'w')
        for k,v in self.ngram_non_zero_map.iteritems():
            cPickle.dump(k,db,-1)
            cPickle.dump(v,db,-1)
        db.close()

        json.dump({'smoothing' : 'context-kneser-ney', 'discount' : self.discount, 'num_db' : num_db, 'den_db' : den_db, 'nz_db' : nz_db, 'order' : self.order, 'unigram_den' : self.unigram_denominator},f)

        flush_and_close(f)

class GoogleCountsModel(LanguageModel):
    """A very very rough backoff model using Google ngrams counts"""
    def __init__(self,ngrams_file,order,timer=None):
        """
        Keyword arguments:

        ngrams_file -- the file with counts from Google (need to document the syntax!!!)
        order -- the order of the model (3 = trigram, 2 = bigram, etc.) (default 3)
        """
        self.ngrams_file = ngrams_file
        self.order = order
        self.ngram_maps = defaultdict(defaultdict_int_factory)
                
        # we do the training which is just loading the counts in memory
        f = open(self.ngrams_file,'r')
        f.readline() # we read the stars
        for i in xrange(1,order+1):
            line = f.readline()
            while line != '*****\n' and line != '':
                tokens = line.split()
                key = self.glue_tokens(tokens[:i])
                val = int(tokens[-1])
                self.ngram_maps[i][key] = val
                line = f.readline()
        self.N = 0
        for v in self.ngram_maps[1].itervalues():
            self.N += v

    # Google uses different tags 
    def tokenize_sentence(self,sentence,order):
        """Returns a list of tokens with the correct numbers of initial and end tags"""
        tokens = sentence.split()
        tokens = ['<S>'] * (order-1) + tokens + ['</S>']
        return tokens

    def train(self,train_corpus,order):
        """This is really not necesary in the case of a model using Google ngrams"""
        pass # already done in the initialization

    def ngram_prob(self,ngram,order):
        try:
            return self.rec_ngram_prob(ngram,order)
        except NameError:
            raise NameError('Negative order for {0}'.format(ngram))
        

    def rec_ngram_prob(self,ngram,order):
        if order < 0:
            raise NameError('Negative order for {0}'.format(ngram))
        real_ngram = ngram[-(order):]
        ngram_key = self.glue_tokens(real_ngram)
        count = self.ngram_maps[order][ngram_key]
        if count != 0:
            if order == 1:
                return float(count) / self.N
            else:
                n_minus_1_gram_key = self.glue_tokens(real_ngram[:-1])
                return float(count) / self.ngram_maps[order-1][n_minus_1_gram_key]
        else:
            return self.rec_ngram_prob(ngram[1:],order-1)

class MultiScoreLanguageModel(LanguageModel):
    """This is a version of LanguageModel that supports more than one generated score for each trigram. STILL UNFINISHED AND POSSIBLY USELESS!!!"""

    def numberOfScores(self):
        """Returns the number of scores calculated by this model"""
        raise NameError('Method numberOfScores not implemented')

    def logprob(self,sentence):
        tokens = self.tokenize_sentence(sentence,self.order)
        return pointwiseFold(lambda x,y : x+y,0,self.numberOfScores(),self.tokens_logprob(tokens,self.order))

    def perplexity(self,text):
        """This method calculates the perplexity score for a text. The text is tokenized by this method (and the necessary <s> and </s> tags added)"""
        return map(lambda x : log_base ** x,self.entropy(text))

    def entropy(self,text):
        """This method calculates the perplexity score for a text. The text is tokenized by this method (and the necessary <s> and </s> tags added)"""
        tokens = self.tokenize_sentence(text,self.order)
        s = [0.] * self.numberOfScores()            
        delta = self.order - 1
        for i in xrange(delta, len(tokens)):
            ng = tokens[i - delta : i + 1]
            ps = self.ngram_prob(ng,self.order)
            for i in xrange(self.numberOfScores()):
                s[i] += -log(ps[i])
        return map(lambda x : x / float(len(tokens) - (self.order - 1)),s)

    def tokens_logprob(self,tokens,order):
        """Returns the logprobs assigned by the model to each token for the specified order. The method skips the first order-1 tokens as initial context."""
        delta = self.order - 1
        return [map(log,self.ngram_prob(tokens[i - delta : i + 1],order)) for i in range(delta,len(tokens))]

    def normalize_logprob_by_sentence_length(self,tokens):
        return map(lambda x : x / len(tokens),pointwiseFold(lambda x,y : x+y,0,self.numberOfScores(),self.tokens_logprob(tokens,self.order)))
    
    def normalize_logprob_by_weighted_sentence_length(self,tokens):
        nums = map(pointwiseFold(lambda x,y : x+y,0,self.numberOfScores(),self.tokens_logprob(tokens,self.order)))
        dens = map(pointwiseFold(lambda x,y : x+y,0,self.numberOfScores(),self.tokens_logprob(tokens,1)))
        res = [0] * self.numberOfScores() 
        for i in xrange(self.numberOfScores()):
            res[i] = nums[i] / (-1 * dens[i])

    def normalized_min_logprob(self,tokens):
        logprobs = self.tokens_logprob(tokens,self.order)
        unigram_logprobs = self.tokens_logprob(tokens,1)
        return min(map(lambda x,y: - (x / y),logprobs,unigram_logprobs))

    # just to avoid having to recalculate the same stuff more than once
    def combined_logprobs(self,tokens):
        """Returns a tuple with raw logprob, logprob normalized by length and logprob normalized by weighted length"""
        logprobs = self.tokens_logprob(tokens,self.order)
        unigram_logprobs = self.tokens_logprob(tokens,1)
        norm_min_logprob = min(map(lambda x,y: - (x / y),logprobs,unigram_logprobs))
        raw_logprob = sum(logprobs)
        n = len(tokens)
        norm_logprob = raw_logprob / n
        w_norm_logprob = raw_logprob / (-1. * sum(unigram_logprobs)) # raw_logprob / ((sum(self.tokens_logprob(tokens,1)) * n) / (self.unigram_mean_probability * n))
        return raw_logprob,norm_logprob,w_norm_logprob,norm_min_logprob
        

