from sklearn.datasets import load_boston
import pandas as pd

boston = load_boston()
print(boston.keys())
print(boston.data.shape)
print(boston.feature_names)

data = pd.DataFrame(boston.data)
data.columns = boston.feature_names
print(" ")
print(data.head())

data['PRICE'] = boston.target

print(data.columns)
print(data.info)
