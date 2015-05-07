# A micro library to work with multisets

import collections
import math
import sys
import numpy as np
from scipy.cluster.vq import kmeans
from scipy.cluster.hierarchy import linkage,fcluster,fclusterdata

class Multiset(collections.defaultdict):
    
    def __init__(self,seq=[]):
        collections.defaultdict.__init__(self,int)
        for el in seq:
            self.add(el)
        
    def add(self,elem):
        self[elem] += 1

    def to_vector(self,space=None):
        """Returns the vector encoding of the multiset.

           Keyword arguments:

           space -- if None the space is considered the ordered sequence of keys, otherwise the multiset is projected in the supplied space (a list of keys)
        """
        if space is None:
            space = self.multiset_minimal_space()

        v = []
        for d in space:
            v.append(self[d])
        return v

    def multiset_minimal_space(self):
        """Returns the multiset minimal space"""
        return sorted(self.keys())

    def to_sequence(self):
        """Transforms the multiset in a sequence"""
        seq = []
        for k,v in self.iteritems():
            seq += [k] * v
        return seq

def vector_sum(v1,v2):
    """Sums two vectors represented as lists"""
    return map(lambda x,y : x + y, v1,v2)

def vector_scalar_multiplication(v,s):
    """Scales the vector by the given number"""
    return map(lambda x : x * s,v)
    
def merge_spaces(s1,s2):
    """Merges two spaces"""
    return sorted(set(s1 + s2))

def centroid_multiset(multisets):
    """Returns the centroid multiset for multisets, a list of multisets"""
    # we compute the common space
    common_space = reduce(merge_spaces,map(lambda x : x.multiset_minimal_space(),multisets))
    # we get the vectors in this space
    vectors = map(lambda x : x.to_vector(common_space),multisets)
    # we calculate the centroid vector
    centroid = vector_scalar_multiplication(reduce(vector_sum,vectors),1./len(vectors))
    # we round the counts
    def round(x):
        i = math.trunc(x)
        if x - i >= 0.5:
            return i + 1
        else:
            return i
    centroid = map(round,centroid)
    # we reconstruct the multiset
    ms = Multiset()
    for i in range(len(common_space)):
        v = centroid[i]
        if v > 0:
            ms[common_space[i]] = v
    return ms

def euclidean_distance(m1,m2):
    """Returns the euclidean distance of two multisets, using their vector encoding"""
    return minkowski_distance(m1,m2,2)

def minkowski_distance(m1,m2,p=1):
    """Returns the generalized Minkowski distance between two multisets, using the vector encoding"""
    s1 = m1.multiset_minimal_space()
    s2 = m2.multiset_minimal_space()
    s = merge_spaces(s1,s2)
    v1 = m1.to_vector(s)
    v2 = m2.to_vector(s)
    res = 0
    for i in range(len(v1)):
        res += math.fabs(v1[i] - v2[i]) ** p
    return res ** (1./p)

def bray_curtis_distance(m1,m2):
    """Return the Bary-Curtis distance of two multisets, using their vector encoding"""
    s1 = m1.multiset_minimal_space()
    s2 = m2.multiset_minimal_space()
    s = merge_spaces(s1,s2)
    v1 = m1.to_vector(s)
    v2 = m2.to_vector(s)
    num = 0
    den = 0
    for i in range(len(v1)):
        num += math.fabs(v1[i] - v2[i])
        den += v1[i] + v2[i]
    return float(num) / den

def binomial_coefficient(n,k):
    return math.factorial(n) / (math.factorial(k) * math.factorial(n - k))

def generate_hierarchical_clusters(bags_of_words_file,t=0.,distance=bray_curtis_distance,method='single',metric='braycurtis'):
    """Performs clustering on the bag of words listed in the input file.

       This function reads a list of bags of words and performs clustering using hierarchical clustering. The bags of words are listed one per line in the input file.
    """

    # first we store the bags in memory
    bags = []
    n_of_bags = 0

    bags_of_words_file.seek(0)
    
    for line in bags_of_words_file:
        bags.append(Multiset(line.split()))
        n_of_bags += 1

    # now we create the condensed distance vector

    dv = np.zeros(binomial_coefficient(n_of_bags,2)) # we create the space in memory first

    i = 0
    dv_index = 0
    for i in range(n_of_bags):
        b1 = bags[i]
        for b2 in bags[(i+1):]:
            dv[dv_index] = distance(b1,b2)
            dv_index += 1

    # now we create the hierarchical clustering

    h = linkage(dv,method,metric)

    # and we flatten it
    clusters = fcluster(h,t)
    # and we return a dict of clusters
    clusters_dict = dict()

    bags_of_words_file.seek(0)
    i = 0
    for line in bags_of_words_file:
        line = line.strip()
        c = clusters[i]
        v = clusters_dict.get(c,[])
        v.append(line)
        clusters_dict[c] = v
        i+=1

    return clusters_dict

def generate_linkage_clusters(bags_of_words_file,t=0.,method='single',metric='braycurtis'):
    """Performs clustering on the bag of words listed in the input file.

       This function reads a list of bags of words and performs clustering using hierarchical clustering, but without using the metrics defined in this module. The bags of words are listed one per line in the input file.
    """
        # first of all we need to read the vocabulary
    # we also count the lines
    voc = set()
    n_of_bags = 0
    
    bags_of_words_file.seek(0)

    for line in bags_of_words_file:
        voc.update(line.split())
        n_of_bags += 1

    # this is the space inside which the multiset vectors live
    space = sorted(voc)

    # we create the numpy array that will store the bags vectors
    data = np.zeros((n_of_bags,len(space)))

    # now we store the bags as vectors in memory
    i = 0

    bags_of_words_file.seek(0)

    for line in bags_of_words_file:
        m = Multiset(line.split())
        data[i] = m.to_vector(space)
        i += 1
        
    # now we can perform the clustering

    clusters = fclusterdata(data,t,metric=metric,method=method)
    
    # and we return a dict of clusters
    clusters_dict = dict()

    bags_of_words_file.seek(0)
    i = 0
    for line in bags_of_words_file:
        line = line.strip()
        c = clusters[i]
        v = clusters_dict.get(c,[])
        v.append(line)
        clusters_dict[c] = v
        i+=1

    return clusters_dict

    return kmeans(data,k)


def pretty_print_clusters_dict(clusters_dict,stream=sys.stdout):
    for k,v in clusters_dict.iteritems():
        stream.write('*** Cluster {0} ***\n'.format(k))
        for x in v:
            stream.write('{0}\n'.format(x))
        stream.write(' - Cluster centroid: "{0}" -\n'.format(' '.join(centroid_multiset(map(lambda x : Multiset(x.split()),v)).to_sequence())))
        stream.write('\n')

def generate_k_means_clusters(bags_of_words_file,k=50):
    """Performs clustering on the bag of words listed in the input file.

       This function reads a list of bags of words and performs clustering using k-means clustering. The bags of words are listed one per line in the input file.
    """
    # first of all we need to read the vocabulary
    # we also count the lines
    voc = set()
    n_of_bags = 0
    
    bags_of_words_file.seek(0)

    for line in bags_of_words_file:
        voc.update(line.split())
        n_of_bags += 1

    # this is the space inside which the multiset vectors live
    space = sorted(voc)

    # we create the numpy array that will store the bags vectors
    data = np.zeros((n_of_bags,len(space)))

    # now we store the bags as vectors in memory
    i = 0

    bags_of_words_file.seek(0)

    for line in bags_of_words_file:
        m = Multiset(line.split())
        data[i] = m.to_vector(space)
        i += 1

    return data
        
    # now we can perform the clustering
    
    return kmeans(data,k)



def test(dist):
    s = ['the cat', 'the very big cat', 'the very very big cat', 'a dog']
    for i in range(len(s)):
        s1 = s[i]
        for s2 in s[(i+1):]:
            m1 = Multiset(s1)
            m2 = Multiset(s2)
            print 'The distance between "{0}" and "{1}" is: {2}'.format(s1,s2,dist(m1,m2))
    
    
    
