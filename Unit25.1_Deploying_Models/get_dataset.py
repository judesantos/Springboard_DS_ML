import pandas as pd

url = 'http://s3.amazonaws.com/assets.datacamp.com/course/Kaggle/train.csv'

df = pd.read_csv(url)

include = ['Age', 'Sex', 'Embarked', 'Survived']

df_ = df[include]

df_.to_csv('train.csv')
