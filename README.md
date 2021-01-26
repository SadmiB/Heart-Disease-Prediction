# Heart Disease Prediction
The current project uses machine learning to predict a heart disease in individuals depending on information we have about them, using information we have about previous individuals.

We will be using Azure Machine Learning and Hyperdrive, aftre comparing we will choose the best model and deploy it to be served as azure endpoints.

## Project Set Up and Installation
*OPTIONAL:* If your project has any special installation steps, this is where you should put it. To turn this project into a professional portfolio project, you are encouraged to explain how to set up this project in AzureML.

## Dataset
### Overview
We will be using a dataset about hear disease from [kaggle](https://www.kaggle.com/ronitf/heart-disease-uci)

* age: age in years 
* sex: (1 = male; 0 = female) 
* cp: chest pain type
* trestbps: resting blood pressure (in mm Hg on admission to the hospital)
* chol: serum cholestoral in mg/dl 
* fbs: (fasting blood sugar &gt; 120 mg/dl) (1 = true; 0 = false) 
* restecg: resting electrocardiographic results 
* thalach: maximum heart rate achieved 
* exang: exercise induced angina (1 = yes; 0 = no) 
* oldpeak: ST depression induced by exercise relative to rest 
* slope: the slope of the peak exercise ST segment
* ca: number of major vessels (0-3) colored by flourosopy 
* thal: 3 = normal; 6 = fixed defect; 7 = reversable defect 
* target: 1 or 0 

### Task
The main task behind this project is to classify individuals whether they have a heart disease or not, we will be using 13 features and the label 'target', the original dataset has more than this, but most of the studies uses these 14 features. 

### Access
In order to access the dataset, I uploaded it to this github repository and I created a dataset in Azure ML using Azure Python SDK.

## Automated ML
*TODO*: Give an overview of the `automl` settings and configuration you used for this experiment
This problem is a classifications problem, hence we using a classification task and the primary metric is Accuracy which we trying to maximize.
We setup the expirement timeout to 20 minutes and the maximum connccurent iterations to 5.
To avoid overfitting, the earlly stopping turned on, the featurization setup to be automatic.

As we saving the ONNX model, we setup compatibity to ONNX mdoels to be True. We avoid the calssification algorithms KNN and LinearSVM in this expirement.

### Results
*TODO*: What are the results you got with your automated ML model? What were the parameters of the model? How could you have improved it?

*TODO* Remeber to provide screenshots of the `RunDetails` widget as well as a screenshot of the best model trained with it's parameters.

## Hyperparameter Tuning
*TODO*: What kind of model did you choose for this experiment and why? Give an overview of the types of parameters and their ranges used for the hyperparameter search


### Results
*TODO*: What are the results you got with your model? What were the parameters of the model? How could you have improved it?

*TODO* Remeber to provide screenshots of the `RunDetails` widget as well as a screenshot of the best model trained with it's parameters.

## Model Deployment
*TODO*: Give an overview of the deployed model and instructions on how to query the endpoint with a sample input.

## Screen Recording
*TODO* Provide a link to a screen recording of the project in action. Remember that the screencast should demonstrate:
- A working model
- Demo of the deployed  model
- Demo of a sample request sent to the endpoint and its response

## Standout Suggestions
*TODO (Optional):* This is where you can provide information about any standout suggestions that you have attempted.

### Saving ONNX model:
I setup AutoML to be compatible with ONNX to make it possible saving the best model in ONNX format. The setps for retreiving the model, saving and testing  it are included in the notebook.
