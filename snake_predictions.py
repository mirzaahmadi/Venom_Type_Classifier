""" 
snake_predictions.pyy
Usage: python snake_predictions.py 'dataset'

This file ultimately takes the 'incomplete_snake_dataset' (which has input protein composition variables but NO output venom type variables) 
and runs it against a trained support vector machine model. The model was trained off the entire dataset 'complete_snake_dataset.csv'. 

The file outputs a list of snake species (from 'incomplete_snake_dataset') and their predicted venom types

afterwards, I can compare the predicted venom type values I got with the actual values (that I collected through literature review), which are 
in the dataset 'ANSWERS_incomplete_snake_dataset.xlsx'

"""

import csv
import sys

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC


def main():
    # Check command-line arguments
    if len(sys.argv) != 2:
        sys.exit("Usage: python snake_predictions.py 'dataset'")

    # Load data from spreadsheet and split into train and test sets
    proteins_training, venom_types_training = load_data(sys.argv[1]) #this data will be the training data to train the model, based off the completed dataset
    testing_dataset = input("Input testing dataset file name: " ) #then, the user will input the testing dataset file name - to be tested with the now trained model
    proteins_to_be_tested = load_data(testing_dataset) #the testing_dataset needs to be put into the load_data() funtion as well, in order to clean the data
    
    # Train model based off of COMPLETED DATASET
    model = train_model(proteins_training, venom_types_training)

    # Make predictions on new dataset
    new_predictions = model.predict(proteins_to_be_tested) #after model is trained, test it with the INCOMPLETE DATASET
    species_names = get_species_names(testing_dataset) #pass it into function to get the names of the snake species, in order to pair up nad present with their predicted venom type
    
    final_predictions = []
    for item in new_predictions: #convert numerical "0 and 1" venom type labelling back to 'neuro' and 'hemo' in order to make it more human readable
        if item == 0: 
            final_predictions.append("Hemotoxic")
        if item == 1:
            final_predictions.append("Neurotoxic")
    
    print()
    print("Predictions based on new dataset: ")
    print()
    #Presents the predictions (of venom types) along with the corresponding species names
    for i, (species_names, final_predictions) in enumerate(zip(species_names, final_predictions)):
        print(f"{species_names} -> {final_predictions}")
        

def get_species_names(t_dataset): #This function simply gets the names of each snake species from the INCOMPLETE DATASET
    snake_species_names = []
    with open(t_dataset, 'r') as file:
        csv_reader = csv.reader(file) #reads it and creates the instance csv_reader
        row = next(csv_reader) #This skips the header row of the dataset
        
        for row in csv_reader:
            names = row[0]
            snake_species_names.append(names)
            
    return snake_species_names
    

def load_data(filename):
    """
    Load shopping data from a CSV file `filename` and convert into a list of
    protein lists and a list of venom type labels. Return a tuple (proteins, venom types).
    """
    protein_list = [] #This will keep a list of all the proteins
    venom_type_list = [] #This will keep a list of all the venom types
    
    converted_protein_list = [] #this will keep a list of all the proteins after being converted to numerical values
    converted_VT_list = [] #This will keep a list of all the venom type values after being converted to numerical values
       
       
    with open (filename, 'r') as file: #opens the filename(csv file) in the read format
        csv_reader = csv.reader(file) #reads it and creates the instance csv_reader
        row = next(csv_reader) #This skips the header row of the dataset
        
        if len(row) == 16: #THIS SECTION WILL RUN WHEN THE COMPLETED DATASET IS PASSED INTO THIS FUNCTION, cleaning it up for training
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
        
        if len(row) == 15: #THIS SECTION WILL RUN WHEN THE INCOMPLETE DATASET IS PASSED INTO THIS FUNCTION, cleaning it up for testing
            for row in csv_reader:
                protein_variables = row[1:14] # for each row, this gives you the values of the columns up to the 14th, excluding the first column (which is just the species names)
                protein_list.append(protein_variables) #append each of the protein_variables to get a list of the lists, by appending them to the protein_list
                
            for list in protein_list:
                converted_protein_list.append(convert_protein_list_components(list))
                
            return converted_protein_list


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


def train_model(x_training_data, y_training_data): #trains model based off the entire dataset of the COMPLETED DATASET
    """
    Given a list of evidence lists and a list of labels, return a
    fitted k-nearest neighbor model (k=1) trained on the data.
    """
    model = SVC() #model used to train is the support vector classifier
    model.fit(x_training_data, y_training_data)
    return model


if __name__ == "__main__":
    main()
    
