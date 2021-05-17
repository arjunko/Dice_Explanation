import dice_ml
from dice_ml.utils import helpers # helper functions
import streamlit as st
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler,OneHotEncoder,LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

df = pd.read_csv(r"C:\adult.csv")
continuous_cols = df.select_dtypes(exclude=["object","category"]).columns
d = dice_ml.Data(dataframe=df, continuous_features=list(continuous_cols), outcome_name='income')
target = df["income"]
#Split data into train and test split
from sklearn.model_selection import train_test_split
X=df.drop(['income'],axis=1)
y= df['income']
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.30,stratify=target)

numerical= continuous_cols
categorical = X_train.columns.difference(numerical)
from sklearn.compose import ColumnTransformer

#We create the preprocessing pipelines for both numeric and categorical data
numeric_transformer = Pipeline(steps=[('scaler',StandardScaler())])
categorical_transformer = Pipeline(steps=[('onehot',OneHotEncoder( handle_unknown='ignore'))])

transformation = ColumnTransformer(transformers=[('num',numeric_transformer,numerical),
                                                 ('cat',categorical_transformer,categorical)])

# Append classifier to preprocessing pipeline.
# Now we have a full prediction pipeline.
clf = Pipeline(steps=[('preprocessor',transformation),
                     ('classifier',LogisticRegression(max_iter=1000))])

model = clf.fit(X_train,y_train)
# provide the trained ML model to DiCE's model object
backend = 'sklearn'
m = dice_ml.Model(model=model, backend=backend)

# initiate DiCE
exp_random = dice_ml.Dice(d, m, method="random")

query_instances = X_train[4:5]
dice_exp_random = exp_random.generate_counterfactuals(query_instances, total_CFs=4, desired_class="opposite",random_seed=0)

dice_vis=dice_exp_random.visualize_as_dataframe(show_only_changes=True)
st.write("Displaying Dice Explanation")
st.dataframe(dice_vis)


