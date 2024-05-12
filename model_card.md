# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details
A random forest modle to predict the incomme of an indivial base on the census infromation.

## Intended Use
To predict whether the incomme of an indivial above or below 50k/yr.

## Training Data
1994 Census Database
https://archive.ics.uci.edu/dataset/20/census+income

80% of the databased used ofr training.

## Evaluation Data
Reserved 20% of the data was used for testing.

## Metrics
Precision, recall and the f1 score were to measure the performance of the model on the evaluation dataset.

- Perisoin on the test set: 0.632919254658385
- Recall rate on the test set: 0.7421704297159505
- F1 score on the test set: 0.6832048273550118

## Ethical Considerations
Participants of the census don't know the purpose, benefits, risks, and funding behind the study.

## Caveats and Recommendations
- Dataset is to old
- Data imputation may help imporve the performance