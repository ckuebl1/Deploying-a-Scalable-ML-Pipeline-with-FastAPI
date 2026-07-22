# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details

This model is a Random Forest Classifier trained using scikit_learn's RandomForestClassifier with default hyperparameters. 
It was developed as part of a project to build a scalable machine learning pipeline with FastAPI. 

## Intended Use

This model is intended for educational purposes only. It is used to predict whether an individual's income exceeds $50,000 based on
US Census data. 

## Training Data

The model was trained on the UCI Census Income Dataset (census.csv), which contains demographic and employment information. This dataset
was split into 80% training data and 20% testing data using scikit_learn's train_test_split function. Categorical functions were encoded
using OneHotEncoder and a LabelBinarizer was used to label salary as either >50k or <= 50k. 

## Evaluation Data

The evaluation/test data is the 20% that was split from the same census.csv as the training data. It was processed using the same OneHotEncoder and LabelBinarizer. 

## Metrics

The model's performance was measured using precision, recall, and F1 score. On the test set, the model achieved:
- Precision: 0.7327
- Recall: 0.6196
- F1 Score: 0.6714

Performance was also computed on slices of data for each categorical feature (slice_output.txt has the full breakdown). Performance varied significantly across slices, for example, recall was noticeably lower for individuals with less education (F1 of 0.4167 for education: 10th) compared to those with much more education (F1 of 0.8877 for education: Prof-school).

## Ethical Considerations

This dataset includes demographic attributes such as race, sex, and native country. The slice-based analysis revealed meaningful disparities in model performance across these groups. 

## Caveats and Recommendations

This model was created using default hyperparameters, so performance could likely be improved with hyperparameter optimization. Some data slices have very small sample sizes (e.g., Married-af-spouse, 3), making their metrics unreliable. Users of this model should be cautious not to generalize based on small slices and further work should focus on collecting a more balanced dataset. 