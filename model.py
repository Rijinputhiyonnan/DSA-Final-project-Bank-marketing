import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.utils import resample
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, fbeta_score, precision_score, f1_score
from sklearn.svm import SVC
import pickle
import pandas as pd
from sklearn.utils import resample

# Load the dataset
df = pd.read_csv('bank-additional-full.csv', delimiter=';')

# Filter rows with 'nonexistent' or 'unknown' values
condition = (df['poutcome'] == 'nonexistent') | (
    (df['job'] == 'unknown') |
    (df['marital'] == 'unknown') |
    (df['education'] == 'unknown') |
    (df['default'] == 'unknown') |
    (df['housing'] == 'unknown') |
    (df['loan'] == 'unknown')
)

print(f"Total number of rows with condition: {condition.sum()}")

# Find rows with two or more 'unknown' or 'nonexistent' values and y == 0
columns_to_check = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'poutcome']
boolean_df = df[columns_to_check].apply(lambda x: x.isin(['unknown', 'nonexistent']))
row_sums = boolean_df.sum(axis=1)
condition = (row_sums >= 2) & (df['y'] == 0)

total_count = condition.sum()
print(f"Total number of rows with two or more 'unknown' or 'nonexistent' values and y == 0: {total_count}")

# Clean the dataframe
df_cleaned_1 = df[~condition]

# Replace 'unknown' and 'nonexistent' with mode values in each column
for c in columns_to_check:
    mode_value = df_cleaned_1[c].mode()[0]
    df_cleaned_1[c] = df_cleaned_1[c].replace(['unknown', 'nonexistent'], mode_value)

# Drop columns
df_cleaned_1.drop(['poutcome', 'nr.employed', 'emp.var.rate', 'cons.conf.idx'], axis=1, inplace=True)

# Drop rows where age > 60 and y == 0
df_cleaned_1.drop(df_cleaned_1[(df_cleaned_1['age'] > 60) & (df_cleaned_1['y'] == 0)].index, inplace=True)

# Drop rows where campaign > 10 and y == 0
df_cleaned_1.drop(df_cleaned_1[(df_cleaned_1['campaign'] > 10) & (df_cleaned_1['y'] == 0)].index, inplace=True)

# Drop rows where duration > 1000 and y == 0
df_cleaned_1.drop(df_cleaned_1[(df_cleaned_1['duration'] > 1000) & (df_cleaned_1['y'] == 0)].index, inplace=True)

# Check the value counts of 'y'
print(df_cleaned_1['y'].value_counts(), "Y counts")

# Filter the majority and minority classes
df_majority = df_cleaned_1[df_cleaned_1['y'] == 'no']
df_minority = df_cleaned_1[df_cleaned_1['y'] == 'yes']
print(df_cleaned_1[df_cleaned_1['y'] == 0].head(), "head of yo")
print(df_majority.shape)
print(df_majority.head())

# Downsample majority class if it's not empty
if not df_majority.empty:
    df_majority_downsampled = resample(df_majority,
                                       replace=True,    # sample with replacement
                                       n_samples=len(df_minority), # to match minority class
                                       random_state=42)
else:
    print("Majority class is empty, cannot downsample.")

# Combine two class
df_bln = pd.concat([df_majority_downsampled, df_minority])
df_bln.head()

# Handling outliers in the 'age' column
median_age = df_bln['age'].median()
Q1 = df_bln['age'].quantile(0.25)
Q3 = df_bln['age'].quantile(0.75)
IQR = Q3 - Q1

df_bln.loc[df_bln['age'] < Q1 - 1.5 * IQR, 'age'] = median_age
df_bln.loc[df_bln['age'] > Q3 + 1.5 * IQR, 'age'] = median_age

# Replace 'pdays' value of 999 with 0 and create a new 'pdays_bin' column
df_bln['pdays'] = df_bln['pdays'].replace(999, 0)
bins = [-1, 5, 10, 20, float('inf')]
labels = ['0-5 days', '6-10 days', '11-20 days', '20+ days']
df_bln['pdays_bin'] = pd.cut(df_bln['pdays'], bins=bins, labels=labels)
df_bln.drop('pdays', axis=1, inplace=True)

# Drop the 'contact' column
df_bln.drop('contact', axis=1, inplace=True)



# Define the custom order for months
month_order = ['feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec', 'jan']

# Create a categorical type with the custom order
df_bln['month'] = pd.Categorical(df_bln['month'], categories=month_order, ordered=True)

# Sort the DataFrame by the custom month order
df_bln = df_bln.sort_values('month')

# Assign quartiles
quartile_labels = ['Q1', 'Q2', 'Q3', 'Q4']
df_bln['quartile'] = pd.qcut(df_bln.index, q=4, labels=quartile_labels)



# Binning 'duration'
bins = [-100, 200, 300, 400, float('inf')]
labels = ['0-100 sec', '101-200 sec', '200-300 sec', '300+ sec']
df_bln['duration_bin'] = pd.cut(df_bln['duration'], bins=bins, labels=labels)
df_bln.drop('duration', axis=1, inplace=True)

# Binning 'campaign'
bins = [-1, 5, 10, float('inf')]
labels = ['0-5 times', '6-10 times', '10+ times']
df_bln['campaign_bin'] = pd.cut(df_bln['campaign'], bins=bins, labels=labels)
df_bln.drop('campaign', axis=1, inplace=True)

# One hot encoding
df_bln = pd.get_dummies(df_bln, columns=['job', 'marital', 'education', 'default', 'housing', 'loan', 'day_of_week', 'pdays_bin', 'duration_bin', 'campaign_bin', 'quartile'])

# Check for non-numeric columns
non_numeric_columns = df_bln.select_dtypes(exclude=['int64', 'float64']).columns
print("Non-numeric columns:", non_numeric_columns)


print(df_bln.head(), "head of balance df")
# Scaling for the columns 'age', 'previous', 'cons.price.idx', 'euribor3m'
scaler = MinMaxScaler()
df_bln[['age', 'previous', 'cons.price.idx', 'euribor3m']] = scaler.fit_transform(df_bln[['age', 'previous', 'cons.price.idx', 'euribor3m']])

# Split the data into train and test sets
x = df_bln.drop('y', axis=1)
y = df_bln['y']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

print(y_train.value_counts(), 'ytrain unique values')
print(y_train.head(), "ytraind head")
# Define and train the SVC model
svc_model_tuned = SVC(C=1, coef0=0.0, degree=3, kernel='poly')
svc_model_tuned.fit(x_train, y_train)

# Save the model using pickle
with open("model_project.pkl", "wb") as model_file:
    pickle.dump(svc_model_tuned, model_file)

# Optional: Print some initial rows for verification
print(x.head(2), y.head(2))
