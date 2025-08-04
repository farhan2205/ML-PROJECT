# House Price Prediction - AWS ML Project

## Objective
Predict house prices using the Boston Housing dataset with a linear regression model.

## Steps
- Data loading and preprocessing
- Model training and evaluation
- Visualization of results
- (Optional) Integration with AWS S3 and SageMaker

## Requirements
- Python 3.x
- pandas, numpy, matplotlib, scikit-learn
- boto3 (for AWS S3 integration)
- AWS account (for SageMaker)

## How to Run
1. Install requirements:  
   `pip install pandas numpy matplotlib scikit-learn boto3`
2. Run the script:  
   `python ml_boston_housing.py`
3. (Optional) Upload dataset to S3 and train on SageMaker.

## Results
The script prints RMSE and R2 score, and plots predicted vs. actual prices.

## AWS Integration (Optional)
- Store the dataset on S3 using boto3.
- Train the model using SageMaker Jupyter Notebook.

## References
- [Boston Housing Dataset](https://scikit-learn.org/stable/datasets/toy_dataset.html#boston-house-prices-dataset)
- [AWS SageMaker](https://aws.amazon.com/sagemaker/)