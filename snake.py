""" 
snake.pyy
Usage: python snake.py 'dataset'

This file splits the complete_snake_dataset.csv into 70% training data and 30% testing data.
The data is trained on a support vector machine model.

After training, the model runs on the testing data, and it outputs how many it got correct and how many it got wrong. 
(We are able to calculate the correct and incorrect scores by comparing the testing prediction data with the actual 
data that is in the complete_snake_dataset.csv for this testing data)
The file also outputs:
A true Neuro rate = the proportion of ACTUAL neuro labels that were accurately identified
A true Hemo rate = the proportion of ACTUAL hemo labels that were accurately identified 

"""

import csv
import sys

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC


TEST_SIZE = 0.30


def main():

    # Check command-line arguments
    if len(sys.argv) != 2:
        sys.exit("Usage: python snake.py 'dataset'")

    # Load data from spreadsheet and split into train and test sets
    proteins, venom_types = load_data(sys.argv[1]) #proteins = the input values, venom_types = the associated output values
    X_train, X_test, y_train, y_test = train_test_split(
        proteins, venom_types, test_size=TEST_SIZE               
    )
    
    # the train_test_split function randomizes the selection of samples for the training and test sets. 
    # By default, it shuffles the data before splitting, ensuring that the samples are randomly distributed between the two sets. 
    # This randomization helps in reducing bias and ensuring that the model learns from a diverse set of examples.
    
    """ 
    The above train_test_split function is built into sci-kit-learn, where it automatically divvys up 
    the data into training groups and testing groups. 
    x train = training group with the associated input values
    y test = testing group with the associated output values
    y train = training group with the associated input values
    y test = testing group with the associated output values
    
    TEST_SIZE (in this case, 30%) is the percentage of data that will be in the training group
    """

    # Train model and make predictions
    model = train_model(X_train, y_train) #training the model based off of the X_training data and the y_training data
    predictions = model.predict(X_test)
    sensitivity, specificity = evaluate(y_test, predictions)

    # Print results
    print(f"Correct: {(y_test == predictions).sum()}")
    print(f"Incorrect: {(y_test != predictions).sum()}")
    print(f"True Neuro Rate: {100 * sensitivity:.2f}%")
    print(f"True Hemo Rate: {100 * specificity:.2f}%")


def load_data(filename):
    """
    Load shopping data from a CSV file `filename` and convert into a list of
    protein lists and a list of venom type labels. Return a tuple of two lists (proteins, venom types).
    """
    protein_list = [] #This will keep a list of all the proteins
    venom_type_list = [] #This will keep a list of all the venom types
    
    converted_protein_list = [] #this will keep a list of all the proteins after being converted to numerical values
    converted_VT_list = [] #This will keep a list of all the venom type values after being converted to numerical values
       
    
    with open (filename, 'r') as file: #opens the filename(csv file) in the read format
        csv_reader = csv.reader(file) #reads it and creates the instance csv_reader
        row = next(csv_reader) #This skips the header row of the dataset
        
        for row in csv_reader:
            protein_variables = row[1:14] # for each row, this gives you the values of the columns up to the 14th, excluding the first column (which is just the species names)
            venom_type_variables = row[15] # for each row, this gives you the values for the 15th row (which are the label columns)
            protein_list.append(protein_variables) #append each of the protein_variables to get a list of the lists, by appending them to the protein_list
            venom_type_list.append(venom_type_variables) #append each of the labelled values to the venom_type_list

        for list in protein_list:
            converted_protein_list.append(convert_protein_list_components(list))
            
        for item in venom_type_list:
            converted_VT_list.append(convert_VT_list_components(item))
        
    return converted_protein_list, converted_VT_list


def convert_protein_list_components(l): #this function converts each of the values in the evidence list to their appropriate numerical components
    new_list = [float(item) for item in l] #iterate through every item in list and convert it to a float
    
    return new_list

def convert_VT_list_components(i): #this function converts each of the items in the VT list to ints
    VT_dict = {
    'Neuro': 1,
    'Hemo': 0
    }
    VT_label = VT_dict[i]
    
    return VT_label


def train_model(evidence, labels): #this (evidence, labels) is x_training data and y_training data
    """
    Given a list of evidence lists and a list of labels, return a
    fitted k-nearest neighbor model (k=1) trained on the data.
    """
    model = SVC() #This line creates a KNeighborsClassifier object with n_neighbors set to 1. This means that the model will consider only the closest neighbor when making predictions.
    model.fit(evidence, labels) #This line fits (trains) the KNN model using the provided evidence (feature vectors) and labels. The fit() method learns the relationships between the features and the corresponding labels from the training data.
    
    return model


def evaluate(labels, predictions): #now, the evaluate method is called, which takes in the y_testing values (labels), and the predictions (the X_testing values) - which are the y_values or labels after running these X_testing values through a model
    """
    `sensitivity` should be a floating-point value from 0 to 1
    representing the "true neuro rate": the proportion of
    actual neuros that were accurately identified.

    `specificity` should be a floating-point value from 0 to 1
    representing the "true hemo rate": the proportion of
    actual hemos that were accurately identified.
    """
    number_of_true_neuros = 0
    number_of_false_neuros = 0
    number_of_true_hemos = 0
    number_of_false_hemos = 0

    for actual, predicted in zip(labels, predictions): #actual, predicted == actual y values, predicted y values
        #So, the values of actual and predicted are going to be either 1 or 0 (True or False)
        #So, I have to use these values to check the proportions of True and Falses, to get the number of true neuros, and the number of true hemos
        
        if actual == 1 and actual == predicted: #if the actual y-value equals NEURO, and it is the same as predicted (prediction is also neuro)
            number_of_true_neuros += 1
        elif actual == 1 and actual != predicted: #if the actual y-value equals NEURO, BUT it is not the same as predicted (prediction is not neuro)
            number_of_false_neuros += 1
        elif actual == 0 and actual == predicted: #if the actual y-value equals HEMO, and it is the same as predicted (prediction is also hemo)
            number_of_true_hemos += 1
        elif actual == 0 and actual != predicted: #if the actual y-value equals HEMO, BUT it is not the same as predicted (prediction is not hemo)
            number_of_false_hemos += 1
            
    sensitivity = float(number_of_true_neuros / (number_of_true_neuros + number_of_false_neuros))
    specificity = float(number_of_true_hemos / (number_of_true_hemos + number_of_false_hemos))
    
    return sensitivity, specificity
    

if __name__ == "__main__":
    main()
