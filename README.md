# Venom Type Classifier

This project utilizes machine learning to predict the venom type of different snake species based solely on the protein composition of their venom. 

**Video Demonstration and Explanation of Algorithms:** https://www.youtube.com/watch?v=-6sjRMZIWcw&t=4s 

## Installation
clone the GitHub repository to your local machine: https://github.com/mirzaahmadi/Venom_Type_Classifier.git

**Requirements:** Python 3.X, matplotlib, sklearn, pandas

## Usage
### PCA_Snake_Data.py: 
First, ensure the snake venom dataset you are using is correctly passed on to the pandas function 'read_csv().' Then, in your terminal, run the following command: '**python PCA_Snake_Data.py**'

### snake.py: 
In your terminal, run the following command: **python snake.py complete_snake_dataset.csv**

### snake_predictions.py: 
In your terminal, run the following command: **python snake.py complete_snake_dataset.csv**. Then, when prompted, input the testing dataset, **incomplete_snake_dataset.csv**. 

## How it's made:
Language: Python | Libraries: sklearn, matplotlib, pandas

Leveraging various machine learning libraries, I created three files that take as input a completed snake venom dataset with the following parameters: snake species names, snake venom proteins, and snake venom types. Using this dataset, the program can visualize the data in a PCA plot, use it to train a machine learning model and evaluate this newly-trained model to make venom type predictions for an incomplete dataset. 

### PCA Visualization
In the file 'PCA_Snake_Data.py' the program takes as input the completed snake venom dataset, 'complete_snake_dataset.csv' which includes species names, protein compositions, and corresponding venom type values (either hemotoxic or neurotoxic). Leveraging matplotlib, the program outputs a PCA plot to visualize the data graphically. 

### Model Training and Testing
In the file 'snake.py' the program takes as input the completed snake venom dataset, 'complete_snake_dataset.csv,' splitting it into randomized training and testing subsets to train and test a support vector machine model, which is created using sklearn. This program outputs the number of correct and incorrect predictions from the testing data, as well as a true neurotoxic rate (the proportion of neurotoxic species that were correctly identified as neurotoxic by the model) and a true hemotoxic rate (the proportion of hemotoxic species that were correctly identified as hemotoxic by the model). 

### Model Evaluation
To further evaluate the model, the file 'snake_predictions.py' takes as input the completed snake venom dataset, 'complete_snake_dataset.csv', and an incomplete snake venom dataset (without labeled venom type values), 'incomplete_snake_dataset.csv.' After a support vector machine is trained using the completed dataset, the model is evaluated by making venom type predictions on the incomplete dataset. This file outputs the species names in the incomplete dataset along with the predicted venom types, and these predictions can be compared to the actual venom type values of each species (sourced from various scientific sources) in the file 'ANSWERS_incomplete_snake_dataset.csv.'


