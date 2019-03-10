from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import auc
from sklearn.metrics import mean_squared_error
from sklearn import linear_model
import data_utils
import pandas as pd


train_X, train_Y, test_X = data_utils.load_data()
X_train, X_valid, Y_train, Y_valid = train_test_split(train_X, train_Y, test_size=0.33, random_state=42)

my_model = XGBClassifier(n_estimators=1000, learning_rate=1)
my_model.fit(X_train, Y_train, early_stopping_rounds=5, eval_set=[(X_valid, Y_valid)], verbose=False)

predictions = my_model.predict_proba(X_valid)[:,0]

result_df = pd.DataFrame(data={"result": predictions, "ground_truth": Y_valid})

result_df = result_df.sort_values(by='ground_truth')

valid_auc = auc(result_df['ground_truth'], result_df['result'])





