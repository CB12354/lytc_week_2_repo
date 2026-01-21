"""
ML1 In-Class
.py file
"""
# %% [markdown]
# [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]
# (https://colab.research.google.com/github.com/UVADS/DS-3021/blob/main/
# 04_ML_Concepts_I_Foundations/ML1_inclass.py#scrollTo=9723a7ee)

# [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]
# (https://colab.research.google.com/github.com/UVADS/DS-3001/blob/main/
# 04_ML_Concepts_I_Foundations/ML1_inclass.ipynb#scrollTo=9723a7ee)
# %%

# %%
# import packages
#from turtle import color
from pydataset import data
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import plotly.express as px
# set the dimension on the images
import plotly.io as pio
pio.templates.default = "plotly_dark" # set dark theme

# %%
iris = data('iris')
iris.head()

# %%
# What mental models can we see from these data sets?
## Could we use each flower's measurements to predict its species? 

# What data science questions can we ask?
## Could a classification algorithm like KNN be used to make the mental model happen?
# %%
"""
Example: k-Nearest Neighbors
"""
# We want to split the data into train and test data sets. To do this,
# we will use sklearn's train_test_split method.
# First, we need to separate variables into independent and dependent
# dataframes.

X = iris.drop(['Species'], axis=1).values  # features
y = iris['Species'].values  # target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=42)
# we can change the proportion of the test size; we'll go with 1/3 for now

# %%
# Now, we use the scikitlearn k-NN classifier
neigh = KNeighborsClassifier(n_neighbors=3)
neigh.fit(X_train, y_train)

# %%
# now, we check the model's accuracy:
neigh.score(X_train, y_train)

# %%
# now, we test the accuracy on our testing data.
neigh.score(X_test, y_test)

# %%
"""
Patterns in data
"""
# Look at the following tables: do you see any patterns? How could a
# classification model point these out?
patterns = iris.groupby(['Species'])
patterns['Sepal.Length'].describe()

# %%
patterns['Sepal.Width'].describe()

# %%
patterns['Petal.Length'].describe()

# %%
patterns['Petal.Width'].describe()

# %%
# scatter plot using plotly
fig = px.scatter_3d(iris, x='Sepal.Length', y='Sepal.Width', z='Petal.Length',
                 color='Species', title='Iris Sepal Dimensions')
fig.show()
# %%
"""
Mild disclaimer
"""
# Do not worry about understanding the machine learning in this example!
# We go over kNN models at length later in the course; you do not need to
# understand exactly what the model is doing quite yet.
# For now, ask yourself:

# 1. What is the purpose of data splitting?
## Training a model on data it has already seen is a bad idea. It will generally predict data it has already seen correctly which can bias results to be appear accurate than they actually would be given a proper test data set.
# 2. What can we learn from data testing/validation?
## You can find out the accuracy of the model for unseen data rather than just the data it is trained on. This is more useful because data in the wild is unknown and likely different from training data.
# 3. How do we know if a model is working?
## It depends on the metric. In this case, it's how accurately it classifies the species of the irises based on the petal and sepal measurements.
# 4. How could we find the model error?
## Is error just (1 - accuracy)? The accuracy I got for the training data that was 1/3 of the full data set and with k=3 was .96 and for the test data 0.98, so the error would be .04 and .02 if that is correct

# If you want, try changing the size of the test data
# or the number of n_neighbors and see what changes!
## Lower amounts of testing data made the test more accurate, but that might not be true for all cases. Changing k from 3 made the model less accurate.