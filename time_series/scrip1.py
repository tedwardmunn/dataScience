import pandas as pd
import numpy as np # necessity as pandas is built on np
from IPython.display import Image,display, HTML # to display images

df1 = pd.read_csv("Train_SU63ISt.csv")
print(" columns " + df1.columns)

display(df1)
