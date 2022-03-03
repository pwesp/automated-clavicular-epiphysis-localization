import numpy as np



def IoU(box_a, box_b):
    """
    Intersection over union of two boxes A and B.
    Boxes must be parametrized as follows: Box = [X1, Y1, X2, Y2]
    
    Returns
    -------
    IoU : float
        Intersection over union
    """
    #print('A:', box_a)
    #print('B:', box_b)
    
    # Extract box parameters for boxes A and B
    x1_a, y1_a, x2_a, y2_a = box_a
    x1_b, y1_b, x2_b, y2_b = box_b
    
    # Check if there is any overlap
    # If there is no overlap on the x-axis, return IoU=0.0
    if x2_b < x1_a or x2_a < x1_b:
        print('No Overlap!')
        return 0.0
    # If there is no overlap on the y-axis, return IoU=0.0
    if y2_b < y1_a or y2_a < y1_b:
        print('No Overlap!')
        return 0.0
    # If any edge should be at an identical position, abort mission
    if x2_b == x1_a or x2_a == x1_b or y2_b == y1_a or y2_a == y1_b:
        print('WARNING: One or more edges of the two boxes are identical! Abort mission!')
        return -1
        
    # It the following code is executed, there should be overlap between boxes A and B
    # Compute overlap on the x-axis
    x = np.min([x2_a, x2_b]) - np.max([x1_a, x1_b])
    
    # Compute overlap on the y-axis
    y = np.min([y2_a, y2_b]) - np.max([y1_a, y1_b])
    
    # Calculate IoU
    intersection_area = x*y
    box_a_area        = (x2_a - x1_a) * (y2_a - y1_a)
    box_b_area        = (x2_b - x1_b) * (y2_b - y1_b)
    union             = box_a_area + box_b_area - intersection_area
    IoU               = float(intersection_area) / float(union)
    #print('A   =', box_a_area)
    #print('B   =', box_b_area)
    #print('I   =', intersection_area)
    #print('U   =', union)
    #print('IoU =', IoU)

    return IoU