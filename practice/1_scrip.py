"""
got the program working but accuracy is not getting above 0.333, maybe an
issue with the target data im not sure
"""
import tensorflow as tf
import pandas as pd
import numpy as np
from IPython.display import display
from sklearn import datasets


iris = datasets.load_iris()
df = pd.DataFrame(data= np.c_[iris['data'], iris['target']], columns= iris['feature_names'] + ['target'])
df1 = pd.DataFrame(iris.data, columns=iris.feature_names, )


print("complete df")
display(df)
print("df head")
display(df.head())
print("df types")
display(df.dtypes)
print("df columns")
display(df.columns)
# print("df1 data")
# display(df1.data)

target = df.target
dataset = tf.data.Dataset.from_tensor_slices((df1.values, target.values))
train_dataset = dataset.shuffle(len(df)).batch(1)

def get_compiled_model():
  model = tf.keras.Sequential([
    tf.keras.layers.Dense(4, activation='relu'),
    # tf.keras.layers.Dense(10, activation='relu'),
    tf.keras.layers.Dense(3, activation='sigmoid')
  ])

  model.compile(optimizer='adam',
                loss='binary_crossentropy',
                metrics=['accuracy'])
  return model

model = get_compiled_model()
model.fit(train_dataset, epochs=200)
# model.evaluate(x_test,  y_test, verbose=2)



# (x_train, y_train), (x_test, y_test) = df1.data
# x_train, x_test = x_train / 255.0, x_test / 255.0
# # Build the tf.keras.Sequential model by stacking layers. Choose an optimizer and loss function for training:
#
#
# model = tf.keras.models.Sequential([
#   tf.keras.layers.Flatten(input_shape=(28, 28)),
#   tf.keras.layers.Dense(128, activation='relu'),
#   tf.keras.layers.Dropout(0.2),
#   tf.keras.layers.Dense(10, activation='softmax')
# ])
#
# model.compile(optimizer='adam',
#               loss='sparse_categorical_crossentropy',
#               metrics=['accuracy'])
# # Train and evaluate the model:
#
#
# model.fit(x_train, y_train, epochs=5)
#
# model.evaluate(x_test,  y_test, verbose=2)
