import numpy as np



def difference(a, b):
    """
    Difference of elements in two numpy arrays, here: difference = {a} w/out {b}
    """
    
    a = set(a)
    b = set(b)
    
    difference = a.difference(b)
    
    if difference:
        difference = np.array(list(difference))
    else:
        difference = np.empty([0], dtype=np.int64)
        
    return difference



def intersection(a, b):
    """
    Intersection of elements in two numpy arrays.
    """
    
    a = set(a)
    b = set(b)
    
    intersection = a.intersection(b)
    if intersection:
        intersection = np.array(list(intersection))
    else:
        intersection = np.empty([0], dtype=np.int64)
    
    return intersection



def union(a, b):
    """
    Union of elements in two numpy arrays.
    """
    
    a = set(a)
    b = set(b)
    
    union = a.union(b)
    if union:
        union = np.array(list(union))
    else:
        union = np.empty([0], dtype=np.int64)
    
    return union