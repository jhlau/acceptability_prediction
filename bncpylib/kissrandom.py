# A simpler and possible better random number generator than the mersenne twister
# documented here http://www0.cs.ucl.ac.uk/staff/d.jones/GoodPracticeRNG.pdf

x = 123456789
y = 987654321
z = 43219876
c = 6543217

def seed(x_s,y_s,z_s,c_s):
    """ Seeds the generator, y_s must NOT be 0 """
    global x, y, z, c
    x = x_s
    y = y_s
    z = z_s
    c = c_s % 698769068 + 1

def rand():
    global x, y, z, c
    x = 314527869 * x + 1234567
    y ^= y << 5
    y ^= y >> 7
    y ^= y << 22
    t = 4294584393L * z + c
    c = t >> 32
    z = t
    return x + y + z
    
def random_real():
    """ Generates a random number in the [0,1) interval """
    return rand() / 4294967296.0

def random_integer(n,m):
    """generates a random integer in the interval [n,m] (m must be > n)"""
    return n + rand() % (1 + m - n)
