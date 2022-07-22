# Customer-Segmentation
# Background
Sentiment analysis, often known as opinion mining, is a natural language processing (NLP) method for identifying the positivity, negativity, or neutrality of data. Businesses frequently do sentiment analysis on textual data to track the perception of their brands and products in customer reviews and to better understand their target market.

![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)
![SciPy](https://img.shields.io/badge/SciPy-%230C55A5.svg?style=for-the-badge&logo=scipy&logoColor=%white)
![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Keras](https://img.shields.io/badge/Keras-%23D00000.svg?style=for-the-badge&logo=Keras&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-%23FF6F00.svg?style=for-the-badge&logo=TensorFlow&logoColor=white)
![Plotly](https://img.shields.io/badge/Plotly-%233F4F75.svg?style=for-the-badge&logo=plotly&logoColor=white)
![Kaggle](https://img.shields.io/badge/Kaggle-035a7d?style=for-the-badge&logo=kaggle&logoColor=white)

# Description
Customer segmentation is the process of grouping customers according to their interests, occupation, age, and gender into subgroups that share certain traits. In order to build a marketing plan that targets the most lucrative sectors, the company must first gather insights into the wants or preferences of its customers.

The dataset for training is around 22,000 and 9,494 for testing.

# How to Run the Project
1. Clone the repository
2. Open the directory
3. Locate the `Model.py` inside the previous directory of the cloned repository
4. Run the file on your chosen IDEs
5. If done correctly, this will generate the results

## Neural Network Model Plot
![model plot](src/output.png)

## Model Performance on The Test Dataset
## Model Training/Test Accuracy plot
![acc](src/acc.png)
## Model Training/Test Loss plot
![Loss](src/loss.png)
## Tensorboard plot
![tensorboard](src/tensorboard.png)
## Classification Report
![class report](src/class_report.png)
## Confusion Matrix display
![confusion_matrix](src/confusion_matrix.png)


# Credits
- [Markdown badges source](https://github.com/Ileriayo/markdown-badges)
- [Dataset source](https://www.kaggle.com/datasets/kunalgupta2616/hackerearth-customer-segmentation-hackathon)

# Extra information
## Correlation value of features
1. job_type: 0.13596388961197914
2. marital: 0.06393741369352811
3. education: 0.07207191785440015
4. default: 0.018498692474409054
5. housing_loan: 0.144442194297714
6. personal_loan: 0.06550522151922447
7. communication_type: 0.14713602417060107
8. month: 0.2713965669912321
9. prev_campaign_outcome: 0.3410607961880476
10. customer_age: 0.892754447498973
11. balance: 0.8925648560685057
12. day_of_month: 0.892754447498973
13. last_contact_duration: 0.8995797389957974
14. num_contacts_in_campaign: 0.892754447498973
15. num_contacts_prev_campaign: 0.8918064903466363
