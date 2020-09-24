Name : Ong Hong Rui

# Instruction for pipeline:

## Before using pipeline make sure to :

1. Fetch data and place into a dataframe

2. Drop null value from dataframe

3. Filter only required features from dataframe as per EDA and assign to X

4. Take only 'price' column and assign into y

5. Do train_test_split to split into train and test set

## PipeLine

**Preprocessing**

1. Round the values of bathrooms column (do a custom transformer for it )

2. Standardize all the values of all columns

**Model**

3. Initalise the machine learning learning model/models

## After making pipeline

1. Fit train set data to pipeline

2. Use pipleline to predict test set

3. Evaluate the model/models using preferred matrics

# Description of pipeline:

From Previous EDA done on the housing sales data, the following 6 features were found to be useful: 'living_room_size','latitude','bathrooms','longitude','lot_size','waterfront'; all of which are numerical data.

The 'bathroom' column was the only column that required preprocessing to ensure that it consists of whole numbers instead of decimals; in the context of bathrooms, it is normally described in whole numbers. Hence, the pipeline was designed to preprocess the data of the 'bathroom' column by rounding it to whole number. It then proceeds to standardising the numerical values of all the features using standard scaler. Finally, it will initiate the chosen model; this model can then be trained to fit the train set. The trained model can then be utilized to predict the test set.

A simple visualisation of the pipeline used is as shown :

           Input data
                |
         Rounding of bathroom
            column values
                |
        Scaling all columns
          values using
          StandardScaler
                |
           Execute model

# Overview of EDA findings

1. Almost all columns have missing values

2. Very noisy data found in 'date' column of data. e.g wrongly spelled months and inconsistency in usage of capital letter.

3. The data in the 'bathroom' column requires rounding to whole numbers as column consist of data points with decimals (eg.0.25 and 1.25).

4. The data in the 'condition' column requires conversion to ensure all data are in the lower case. Without this conversion, data such as "fair" and "FAIR" would be counted as separate conditions when they should be counted as the same condition.

5. Based on prototyping with linear models (LASSOCV and linear regressor), the dataset yielded low R-squared values (<60%), hence indicating that they share a non-linear relation with house pricing. Prototyping with non-linear model (Random forest regressor) supports this finding.

6. Out of all the features, 6 were found to be the most influential factors. Hence only these 6 were used for machine learning.

# Explanation of Choice of model

## Regressor Model

For regressor model, Random Forest Regressor was chosen based on the findings in EDA where protype testing was done using linear model (linear regressor,lassoCV) and non-linear model(Random Forest Regressor) as mentioned. Hence Random Forest Regressor was chosen as the machine learning model for the regressor.

## Classifier Model

For Classifier Model, Decision tree classifier and Random Forest Classifier was utilised as these are two very commonly used classifier model.

# Evaluation of Model

## Regressor

For regressor model, Root Mean Square Error (RMSE) and R-square scores were chosen as the main matrix of comparison. Both indicates how close the predictions of the model are as compared to the actual data. Based on the results from EDA, Random Forst Regressor is the best model.

## Classifier

For classifier model , F1 micro score was used as the main evaluation criteria since it is able to address imbalance classes; for the data set, the imbalance was due to the 'medium' class having much more data as compared to the other two classes ('low' and 'high). This could cause the result to be skewed. Hence, F1 micro score was chosen as the evaluation criteria over accuracy score which do not address imbalance in classes.

# Other Considerations for models deployed

1. Computational time. In which, modules or model that requires long computational time such as Recursive feature elimination (RFE) and neural network respectively were not used for machine learning model pipeline in this test.

2. Complexity of the model. Complex models such as neural network requires more time and attempts for obtaining a good model. For the case of neural network, it hyperparameters tuning is required. Due to the varierty of neural network available (e.g. feed forwards model, convolution neural network and multilayer neural network) and the various associated algorithms (e.g. radial basis function and gradient descent), it is challenging to conduct a proper evalutaion. Hence, complex models such as neural network was not evaluated in this test.

# Overall Evaluation of approach to the task

Although both regressor and classifier yielded similar prediction accuracy of ~80%, I am inclined to chose the classifier model. Base on my understanding, house pricing comes in ranges. As such, the classifier would be a more appropriate model to aid housing agents in gauging the price of a house. A regressor model would provide a single price point and would require prior experience or external inputs to aid housing agents in gauging house pricing.
