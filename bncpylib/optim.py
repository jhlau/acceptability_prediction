# optimization algorithms

import bncpylib.random as rand

minus_infinity = - float('inf')

def univariate_multiple_restarts_hill_climbing(f,lowerBound,upperBound,step,iterations,firstGuess=None,seed=34523):
    """Multiple restarts hill climbing for numerical function of a single variable

    f is the function to be optimized
    lowerBound is the lower bound of the search space (the domain of the function)
    upperBound is the upper bound of the search space
    step is the discrete step used to explore the space
    iterations is the number of restart to be used
    seed is the number used to seed the random number generator used for the restarts
    """
    rand.init_genrand(seed)
    if firstGuess:
        global_best = best = firstGuess
    else:
        global_best = best = rand.random_real(lowerBound,upperBound)
    global_best_value = best_value = f(best)
    for i in xrange(iterations):
        while True:
            left = best - step
            right = best + step
            left_value = right_value = minus_infinity
            if left >= lowerBound:
                left_value = f(left)
            if right <= upperBound:
                right_value = f(right)
            if right_value > best_value:
                best = right
                best_value = right_value
            elif left_value > best_value:
                best = left
                best_value = left_value
            else:
                break
        if best_value > global_best_value:
            global_best = best
        # random restart
        best = rand.random_real(lowerBound,upperBound)
        if i < iterations - 1:
            best_value = f(best)
    return global_best
            
def multivariate_multiple_restarts_hill_climbing(f,n,lowerBounds,upperBounds,basis,iterations,firstGuess=None,seed=34523):
    """Multiple restarts hill climbing for numerical function with multiple free parameters

    f is the function to be optimized, f should take a single list (the parameters we are trying to optimize) as input
    n is the number of parameters of the function
    lowerBounds are the lower bounds of the search space for each dimension (a list of length n)
    upperBound are the upper bounds of the search space for each dimension (a list of length n)
    basis is the basis of the space (a list of length n of lists of length n, e.g. for a 3-dimensional space something like [[0.01,0.,0.],[0.,0.01,0.],[0.,0.,0.01]])
    iterations is the number of restart to be used
    seed is the number used to seed the random number generator used for the restarts
    """
    rand.init_genrand(seed)
    if firstGuess:
        global_best = best = firstGuess
    else:
        global_best = best = map(lambda i : rand.random_real(lowerBounds[i],upperBounds[i]),range(n))
    global_best_value = best_value = f(best)
    for i in xrange(iterations):
        while True:
            new_points = [bounded_vector_sum(best,scalar_product(k,step)) for step in basis for k in [1.,-1.]]
            changed = False
            for new_point in new_points:
                new_value = f(new_point)
                if new_value > best_value:
                    changed = True
                    best = new_point
                    best_value = new_value
                    break
            if not changed:
                break
        if best_value > global_best_value:
            global_best = best
        best = map(lambda i : rand.random_real(lowerBounds[i],upperBounds[i]),range(n))
        if i < iterations - 1:
            best_value = f(best)
    return global_best

def scalar_product(s,v):
    return [s * x for x in v]
    
def vector_sum(a,b):
    return [x + y for (x,y) in zip(a,b)]

def bounded_vector_sum(a,b,lowerBounds,upperBounds):
    return [min(max(x+y,l),u) for (x,y,l,u) in zip(a,b,lowerBounds,upperBounds)  ]


# Probably a better option is to just use the optimize module of scipy
