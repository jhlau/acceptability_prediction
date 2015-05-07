# The thing that most likely is going to make replication of the
# experiments difficult is the random number generator. To have
# uniform results across machines, OSs and python versions we should
# use a rng that depends only on the seed. For now we can use this
# version of the Mersenne Twister which is a direct port of the
# original reference C code by Matsumoto and Nishimura. This was found
# online on stackoverflow at:
# http://stackoverflow.com/questions/2469031/open-source-implementation-of-mersenne-twister-in-python

# Fri Oct 19 10:41:08 CEST 2012
# Modified the original port to match an update reference version
# by Makoto Matsumoto


## a C -> python translation of MT19937, original license below ##

##  A C-program for MT19937: Real number version
##    genrand() generates one pseudorandom real number (double)
##  which is uniformly distributed on [0,1]-interval, for each
##  call. sgenrand(seed) set initial values to the working area
##  of 624 words. Before genrand(), sgenrand(seed) must be
##  called once. (seed is any 32-bit integer except for 0).
##  Integer generator is obtained by modifying two lines.
##    Coded by Takuji Nishimura, considering the suggestions by
##  Topher Cooper and Marc Rieffel in July-Aug. 1997.

##  This library is free software; you can redistribute it and/or
##  modify it under the terms of the GNU Library General Public
##  License as published by the Free Software Foundation; either
##  version 2 of the License, or (at your option) any later
##  version.
##  This library is distributed in the hope that it will be useful,
##  but WITHOUT ANY WARRANTY; without even the implied warranty of
##  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
##  See the GNU Library General Public License for more details.
##  You should have received a copy of the GNU Library General
##  Public License along with this library; if not, write to the
##  Free Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA
##  02111-1307  USA

##  Copyright (C) 1997 Makoto Matsumoto and Takuji Nishimura.
##  Any feedback is very welcome. For any question, comments,
##  see http://www.math.keio.ac.jp/matumoto/emt.html or email
##  matumoto@math.keio.ac.jp

## Additional license for the new version

# # A C-program for MT19937, with initialization improved 2002/1/26.
# #    Coded by Takuji Nishimura and Makoto Matsumoto.

# #    Before using, initialize the state by using init_genrand(seed)  
# #    or init_by_array(init_key, key_length).

# #    Copyright (C) 1997 - 2002, Makoto Matsumoto and Takuji Nishimura,
# #    All rights reserved.                          

# #    Redistribution and use in source and binary forms, with or without
# #    modification, are permitted provided that the following conditions
# #    are met:

# #      1. Redistributions of source code must retain the above copyright
# #         notice, this list of conditions and the following disclaimer.

# #      2. Redistributions in binary form must reproduce the above copyright
# #         notice, this list of conditions and the following disclaimer in the
# #         documentation and/or other materials provided with the distribution.

# #      3. The names of its contributors may not be used to endorse or promote 
# #         products derived from this software without specific prior written 
# #         permission.

# #    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# #    "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# #    LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
# #    A PARTICULAR PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# #    CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# #    EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# #    PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# #    PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
# #    LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
# #    NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# #    SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


# #    Any feedback is very welcome.
# #    http://www.math.sci.hiroshima-u.ac.jp/~m-mat/MT/emt.html
# #    email: m-mat @ math.sci.hiroshima-u.ac.jp (remove space)

import copy
import sys

# Period parameters
N = 624
M = 397
MATRIX_A = 0x9908b0dfL   # constant vector a
UPPER_MASK = 0x80000000L # most significant w-r bits
LOWER_MASK = 0x7fffffffL # least significant r bits

mt = []   # the array for the state vector
mti = N+1 # mti==N+1 means mt[N] is not initialized

# initializing the array with a NONZERO seed
def init_genrand(seed):
  # setting initial seeds to mt[N] using
  # the generator Line 25 of Table 1 in
  # [KNUTH 1981, The Art of Computer Programming
  #    Vol. 2 (2nd Ed.), pp102]

  global mt, mti

  mt = []

  mt.append(seed & 0xffffffffL)
  for i in xrange(1, N + 1):
#    mt.append((69069 * mt[i-1]) & 0xffffffffL)
# checking on Matsumoto's homepage I noticed that he has updated the code
# for a problem with the previous initialization routine
# I'm aligning the python code to the new version
     mt.append(1812433253L * (mt[i - 1] ^ (mt[i - 1] >> 30)) + i)
     mt[i] &= 0xffffffffL
     
  mti = i
# end sgenrand

def init_by_array(init_key,key_length):
    global mt, mti
    init_genrand(19650218L)
    i=1
    j=0
    if N > key_length:
        k = N
    else:
        k = key_length
    while k > 0:
        mt[i] = (mt[i] ^ ((mt[i-1] ^ (mt[i-1] >> 30)) * 1664525L)) + init_key[j] + j # non linear
        mt[i] &= 0xffffffffL; # for WORDSIZE > 32 machines
        i+=1
        j+=1
        if i >= N:
            mt[0] = mt[N-1]
            i=1
        if j >= key_length:
            j=0
        k-=1
    k = N-1
    while k > 0:
        mt[i] = (mt[i] ^ ((mt[i-1] ^ (mt[i-1] >> 30)) * 1566083941L)) - i # non linear
        mt[i] &= 0xffffffffL # for WORDSIZE > 32 machines 
        i+=1
        if i>=N:
            mt[0] = mt[N-1]
            i=1
        k-=1
    mt[0] = 0x80000000L # MSB is 1; assuring non-zero initial array

def genrand_int32():
  global mt, mti

  mag01 = [0x0L, MATRIX_A]
  # mag01[x] = x * MATRIX_A  for x=0,1
  y = 0

  if mti >= N: # generate N words at one time
    if mti == N+1:   # if sgenrand() has not been called,
      init_genrand(5489L) # a default initial seed is used

    for kk in xrange((N-M) + 1):
      y = (mt[kk]&UPPER_MASK)|(mt[kk+1]&LOWER_MASK)
      mt[kk] = mt[kk+M] ^ (y >> 1) ^ mag01[y & 0x1]

    for kk in xrange(kk, N):
      y = (mt[kk]&UPPER_MASK)|(mt[kk+1]&LOWER_MASK)
      mt[kk] = mt[kk+(M-N)] ^ (y >> 1) ^ mag01[y & 0x1]

    y = (mt[N-1]&UPPER_MASK)|(mt[0]&LOWER_MASK)
    mt[N-1] = mt[M-1] ^ (y >> 1) ^ mag01[y & 0x1]

    mti = 0

  y = mt[mti]
  mti += 1
  y ^= (y >> 11)
  y ^= (y << 7) & 0x9d2c5680L
  y ^= (y << 15) & 0xefc60000L
  y ^= (y >> 18)

  return y

def genrand_int31():
    """ generates a random number on [0,0x7fffffff]-interval """
    return long(genrand_int32() >> 1)

def genrand_real1():
    """ generates a random number on [0,1]-real-interval """
    return genrand_int32() * (1.0/4294967295.0)

def genrand_real2():
    """ generates a random number on [0,1)-real-interval """
    return genrand_int32()*(1.0/4294967296.0)

def genrand_real3():
    """ generates a random number on (0,1)-real-interval """
    return ((float(genrand_int32())) + 0.5)*(1.0/4294967296.0)

def genrand_res53():
    """ generates a random number on [0,1) with 53-bit resolution """
    a = genrand_int32()>>5
    b = genrand_int32()>>6
    return (a*67108864.0+b)*(1.0/9007199254740992.0)

def random_int(n,m):
  """generates a random integer in the interval [n,m] (m must be > n)"""
  return n + genrand_int32() % (1 + m - n)

def random_element(seq):
  """picks a random element from a sequence"""
  return seq[random_int(0,len(seq)-1)]

def random_real(lower,upper):
  """generates a random real number in the interval [lower,upper]"""
  return lower + (upper - lower) * genrand_real1()

# reservoir sampling
def reservoir_sampling(k,n):
  """Samples k indices from n possible indices, without repetitions.
     The indices are 1 based!

     We use this approach given that we may have to select a large number of indices from a VERY large list of indices. Trying to fit this stuff in memory would be just impossible. We still have to fit a possibly large list of indices in memory though.
  """
  reservoir = range(1,k+1)
  for i in xrange(k+1,n+1):
    j = random_int(1,i)
    if j <= k:
      reservoir[j-1] = i
  return reservoir

def sample(k,l):
    """Samples k elements without repetition from a list of things"""
    l_tmp = copy.copy(l)
    for i in xrange(len(l_tmp) - 1,0,-1):
        j = random_int(0,i)
        tmp = l_tmp[i]
        l_tmp[i] = l_tmp[j]
        l_tmp[j] = tmp
    return l_tmp[:k]

def swap_within_window(seq,window,guarantee_different=False,hard_limit=None):
  """Swaps two elements in seq whose distance is at maximum window.

     The swap is done in place, i.e. the function is destructive. The function
     returns the updated sequence.
     If guarantee_different is set to True the operation is repeated until a different sequence is generated.
  """

  if hard_limit is None:
    hard_limit = sys.getrecursionlimit() - 10

  if hard_limit == 0:
    return seq
  
  l = len(seq)-1

  if l <= 0: # if there are 1 or 0 elements we don't do anything
    return seq
  
  i = random_int(0,l)
  j = -1
  # here we keep on generating a j until it's ok
  while j < 0 or j > l or j == i:
    j = random_int(i-window,i+window)

  if guarantee_different:
    old_seq = copy.copy(seq)
    
  tmp = seq[i]
  seq[i] = seq[j]
  seq[j] = tmp

  if guarantee_different and seq == old_seq:
    return swap_within_window(seq,window,guarantee_different,hard_limit-1)
  else:
    return seq
  

def swap_adjacents(seq):
  """A shortcut for swap_within_window(seq,1)"""
  return swap_within_window(seq,1)

def swap_elements_at_distance_k(seq,k,guarantee_different=False,hard_limit=None):
  """Swaps two elements (selected randomly) that are precisely at distance k. If k is larger than the length it is reduced to len(seq) - 1. The swap is destructive!

     If guarantee_different is set to True the operation is repeated until a different sequence is generated.
  """

  if hard_limit is None:
    hard_limit = sys.getrecursionlimit() - 10

  if hard_limit == 0:
    return seq

  signs = [-1,1]
  l = len(seq)-1

  if l <= 0: # if there are 1 or 0 elements we don't do anything
    return seq
  
  if k > l:
    k = l

  i = -1
  j = -1

  while j < 0 or j > l:
    i = random_int(0,l)
    j = random_element(signs) * k + i

  if guarantee_different:
    old_seq = copy.copy(seq)

  tmp = seq[i]
  seq[i] = seq[j]
  seq[j] = tmp

  if guarantee_different and seq == old_seq:
    return swap_elements_at_distance_k(seq,k,guarantee_different,hard_limit-1)
  else:
    return seq


class WeightedItem(object):
  def __init__(self,item,weight):
    self.item = item
    self.weight = weight


def sample_from_weighted_list(seq):
  """ Samples an object from a list of WeightedItem objects. The weights of the object must add up to 1. The function returns a WightedItem"""

  p = genrand_real2()

  cumulative_prob = 0.0

  for item in seq:
    cumulative_prob += item.weight
    if p < cumulative_prob:
      return item

  return seq[-1] # this should be unreachable but we keep just to be sure
  
# With respect to the reference output there are some small discrepancies
# but the vast majority of the output is ok. I guess they are due to differences
# in the representations of long numbers in C and python. If I have time I'll look
# into this problem

def test(output_stream):
    init = [0x123L, 0x234L, 0x345L, 0x456L]
    length = 4
    init_by_array(init,length)
    output_stream.write('1000 outputs of genrand_int32()\n')
    for i in range(1000):
        output_stream.write('{0:10} '.format(genrand_int32()))
        if i % 5 == 4:
            output_stream.write('\n')
    output_stream.write('1000 outputs of genrand_real2()\n')
    for i in range(1000):
        output_stream.write('{0:.8f} '.format(genrand_real2()))
        if i % 5 == 4:
            output_stream.write('\n')

if __name__ == '__main__':
    test(sys.stdout)
