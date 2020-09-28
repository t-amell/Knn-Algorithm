# Author: Trevor Amell
# Adopted From https://machinelearningmastery.com/tutorial-to-implement-k-nearest-neighbors-in-python-from-scratch/
# Class: CSI-480
# Certification of Authenticity:
# I certify that this is entirely my own work, except where I have given fully documented
# references to the work of others.  I understand the definition and consequences of
# plagiarism and acknowledge that the assessor of this assignment may, for the purpose of
# assessing this assignment reproduce this assignment and provide a copy to another member
# of academic staff and / or communicate a copy of this assignment to a plagiarism checking
# service(which may then retain a copy of this assignment on its database for the purpose
# of future plagiarism checking).

import math


# Calculate Euclidean Distance between 2 vectors
# Euclidean Distance = sqrt(sum i to N (x1_i – x2_i)^2)
def euclidean_distance(row1, row2):
    distance = 0.0

    # Loop Through All Columns For Each Item/Row
    for i in range(len(row1)-1):
        distance += (row1[i] - row2[i])**2
    return math.sqrt(distance)


# Locate The Most Similar Neighbors
def get_neighbors(train, test_row, num_neighbors):
    distances = []
    # Get Euclidean Distance From Every Row Compared To New Test Row
    for train_row in train:
        cur_dist = euclidean_distance(test_row, train_row)
        distances.append((train_row, cur_dist))
    # Sort By Second Item In Tuple (Distance)
    distances.sort(key=lambda tup: tup[1])

    neighbors = []
    # Return num_neighbors Closest Rows
    for i in range(num_neighbors):
        neighbors.append(distances[i][0])
    return neighbors


# Make A Classification Prediction With Neighbors
def predict_classification(train, test_row, num_neighbors):
    neighbors = get_neighbors(train, test_row, num_neighbors)
    output_values = [row[-1] for row in neighbors]
    prediction = max(set(output_values), key=output_values.count)
    return prediction
