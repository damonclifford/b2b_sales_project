import pandas as pd
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer, RobustScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline, FeatureUnion
from datasets import get_data




def get_categorical_features():
  X, y = get_data('../data/trainDF.csv')
  dtypes = pd.DataFrame(X.dtypes.rename('type')).reset_index().astype('str')
  numeric = dtypes[(dtypes.type.isin(['int64', 'float64']))]['index'].values
  categorical = dtypes[~(dtypes['index'].isin(numeric)) & (dtypes['index'] != 'y')]['index'].values
  return categorical


def get_numeric_features():
  X, y = get_data('../data/trainDF.csv')
  dtypes = pd.DataFrame(X.dtypes.rename('type')).reset_index().astype('str')
  numeric = dtypes[(dtypes.type.isin(['int64', 'float64']))]['index'].values
  return numeric


NUM_FEAT = get_numeric_features()
CAT_FEAT = get_categorical_features()


def get_categorical_pipeline():
  # Create the transformers for categorical features
  cat_ct = ColumnTransformer([('categoricals', 'passthrough', CAT_FEAT)])

  # Create the pipeline to transform categorical features
  cat_pipeline = Pipeline([
    ('cat_ct', cat_ct),
    ('ohe', OneHotEncoder(handle_unknown='ignore'))
  ])

  return cat_pipeline

def get_numeric_pipeline():
  # Create the transformers for numeric features
  num_ct = ColumnTransformer([('numerics', 'passthrough', NUM_FEAT)])

  # Create the pipeline to transform numeric features
  num_pipeline = Pipeline([
    ('num_union', num_ct),
    ('scaler', RobustScaler())
  ])
  
  return num_pipeline

def get_pipeline(cat_pipeline, num_pipeline):
  # Create the categorical and numeric pipelines
  #cat_pipeline = get_categorical_pipeline()
  #num_pipeline = get_numeric_pipeline()

  # Create the feature union of categorical and numeric attributes
  ft_union = FeatureUnion([
    ('cat_pipeline', cat_pipeline),
    ('num_pipeline', num_pipeline)
  ])

  pipeline = Pipeline([
    ('ft_union', ft_union)
  ])

  return pipeline

def baseline_model_predictions(X, y, n_targeted):
  # Get all of the instances where the previous campaign was a success
  success = X[X.poutcome == 'success']
  
  # Calcuate how many more instances we need
  n_rest = n_targeted - len(success)
  
  # Randomly choose from the remaining instances
  rest = X[~(X.index.isin(success.index))].sample(n=n_rest, random_state=1)
  
  # Combine the targeted and random groups
  baseline_targets = pd.concat([success, rest], axis=0)
  baseline_ys = y.loc[baseline_targets.index]

  return baseline_ys