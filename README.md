# emotion-text-classification
Text emotion classification using TF-IDF and Naive Bayes (Python, scikit-learn)
# goal
predict the emotion (surprise, anger, joy) expressed in a text

# data
Emotions dataset from Kaggle (https://www.kaggle.com/datasets/abhrajaiswal/emotions-detection-text-dataset)

# methods
- exploratory data analysis (text length, class distribution)
- text vectorization with TF-IDF
- multinomial Naive Bayes classifier (scikit-learn)

# results
 - accuracy: 66.93%
 - main finding: Due to the lack of training data for some emotions (e.g. surprise), we observed a high error rate on the corresponding test data. 

# how to run
pip install -r requirements.txt

# tools
python, pandas, matplotlib, scikit-learn
