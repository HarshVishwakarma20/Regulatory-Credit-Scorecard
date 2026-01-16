import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score,roc_curve
from sklearn.linear_model import LogisticRegression
import pickle

try:
    df = pd.read_csv("Loan.csv",low_memory = False)
    print(f"Data Loaded Successfully. Shape: {df.shape}")
except FileNotFoundError:
    print("Error: loan.csv not found. Please download it from Kaggle.")

valid_status = ['Fully Paid', 'Charged Off', 'Default', 'Does not meet the credit policy. Status:Fully Paid', 'Does not meet the credit policy. Status:Charged Off']
df = df[df['loan_status'].isin(valid_status)].copy()

#defining target variable
df['target'] = np.where(df['loan_status'].str.contains('Charged Off|Default'), 1, 0)

print(f"Filtered Shape: {df.shape}")
print("Target Distribution:\n", df['target'].value_counts(normalize=True))

#target imbalance
plt.figure(figsize=(6,4))
sns.countplot(x='target', hue = 'target',data=df, palette=['green', 'red'],legend = False)
plt.title('Loan Status Distribution (0: Good, 1: Bad)')
plt.xlabel('Loan Status')
plt.ylabel('Count')
#percentage
total = len(df)
for p in plt.gca().patches:
    height = p.get_height()
    plt.gca().text(p.get_x()+p.get_width()/2., height + 500, 
                   '{:1.2f}%'.format(100*height/total), ha="center")
plt.show()

#ids of no use
drop_ids = ['id', 'member_id', 'url', 'desc', 'policy_code']
df.drop(columns=[c for c in drop_ids if c in df.columns], inplace=True)

#Columns Generated After loan approval
leakage_cols = [
    'recoveries', 'collection_recovery_fee', 'total_rec_prncp', 'total_rec_int', 
    'total_pymnt', 'total_pymnt_inv', 'last_pymnt_d', 'last_pymnt_amnt', 
    'next_pymnt_d', 'out_prncp', 'out_prncp_inv'
]
df.drop(columns=[c for c in leakage_cols if c in df.columns], inplace=True)

#dropping columns with 80% missing values as they are useless
limit = len(df) * 0.8
df.dropna(axis=1, thresh=limit, inplace=True)

#splitting early for maximum throughput
X = df.drop(['target', 'loan_status'], axis=1)
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print(f"Final Train Shape: {X_train.shape}")
print(f"Final Test Shape: {X_test.shape}")

#defining woe and iv function
def woe_iv(df,feature,target):
    list=[]
    total_good = df[target].value_counts()[0]
    total_bad = df[target].value_counts()[1]
    
    for value,group in df.groupby(feature,observed = False):
        n_good = group[target].value_counts().get(0,0)
        n_bad = group[target].value_counts().get(1,0)
        #To avoid division by 0
        if n_good == 0:n_good = 0.5
        if n_bad == 0:n_bad = 0.5
            
        good_distribution = n_good/total_good
        bad_distribution = n_bad/total_bad
        woe = np.log(good_distribution/bad_distribution)
        info_val = (good_distribution - bad_distribution)*woe

        list.append({'Feature':feature,'Bin':value,'WoE':woe,'IV':info_val})
    return pd.DataFrame(list)

#Execution
train_data = pd.concat([X_train,y_train],axis = 1)
numerical_col = X_train.select_dtypes(include=np.number).columns
categorical_col = X_train.select_dtypes(include = ['object','category']).columns
binning_process = {}

#list and dictionary to store our transformation maps and list.
info_val_list = []
woe_map = {}
print("Staring Feature engineering and may take time!!")
#processing categorical column
for col in categorical_col:
    train_data[col] = train_data[col].fillna('Missing')
    if train_data[col].nunique() > 50:
        print(f"Skipping {col} (too many categories)")
        continue

    woe_df = woe_iv(train_data,col,'target')
    info_val_list.append({'Feature': col, 'IV': woe_df['IV'].sum()})
    woe_map[col] = woe_df
#processing numerical column
for col in numerical_col:
    try:
        train_data[col+'_bin'],bin_edges = pd.qcut(train_data[col],q=10,duplicates = 'drop',retbins=True)
        woe_df = woe_iv(train_data,col+'_bin','target')
        info_val_list.append({'Feature':col,'IV':woe_df['IV'].sum()})
        woe_map[col] = woe_df
        binning_process[col] = bin_edges
    except Exception as e:
        continue  #Columns may be constant or fail to bin
iv_report = pd.DataFrame(info_val_list).sort_values(by='IV',ascending = False)
print("Top 10 Predictive Features\n")
print(iv_report.head(10))
#dropping high IV columns which are not useful and can cause disruption in model
#also dropping grade,sub_grade,int_rate as they are not raw data and dropping to build a pure model
leakage_drop = [
    'last_fico_range_low',
    'last_fico_range_high',
    'debt_settlement_flag',
    'sub_grade',
    'grade',
    'int_rate',
    'total_rec_late_fee',
    'fico_range_high' #low and high are the same data so no use to keep both
]
cleaned_iv_report = iv_report[~iv_report['Feature'].isin(leakage_drop)]
print("Top 10 Predictors")
print(cleaned_iv_report.head(10))

#woe trend
top_feature = cleaned_iv_report.iloc[0]['Feature'] 
woe_df = woe_map[top_feature]
#sorting
if 'Bin' in woe_df.columns and woe_df['Bin'].dtype.name == 'category':
    woe_df = woe_df.sort_values(by='Bin')
plt.figure(figsize=(10,5))
sns.pointplot(x='Bin', y='WoE', data=woe_df, color='blue', markers='o')
plt.title(f'Weight of Evidence (WoE) Trend: {top_feature}')
plt.xlabel('Bins / Categories')
plt.ylabel('WoE (Log Odds)')
plt.xticks(rotation=45)
plt.grid(True, alpha=0.3)
plt.axhline(0, color='gray', linestyle='--')
plt.show()

#Using >0.02 as cutoff
final_features = cleaned_iv_report[cleaned_iv_report['IV']>0.02]['Feature'].tolist()
print(f"Final Feature Selection : {len(final_features)} features")
print(f"Features: {final_features}")

X_train_woe = pd.DataFrame(index = X_train.index)
print("Transforming all the data into WOE")

for col in final_features:
    if col in woe_map:
        mapping = woe_map[col]
        #mapping categorical columns directly
        if col in categorical_col:
            woe_dict = pd.Series(mapping.WoE.values,index = mapping.Bin).to_dict()
            X_train_woe[col] = train_data[col].map(woe_dict)
        else:
        #mapping the bins in the numerical columns
            try:
                woe_dict = pd.Series(mapping.WoE.values,index = mapping.Bin).to_dict()
                X_train_woe[col] = train_data[col+'_bin'].map(woe_dict)
            except KeyError:
                #For safety purpose
                print(f"Skipping {col} due to mapping error.")
                continue

X_train_woe = X_train_woe.astype(float)
X_train_woe.fillna(0,inplace = True)

print("Training the Logistic Regression Model")
model = LogisticRegression(solver='lbfgs', C=1.0, max_iter=1000)
model.fit(X_train_woe, y_train)

probs_train = model.predict_proba(X_train_woe)[:,1]
auc = roc_auc_score(y_train,probs_train)
gini_coef = 2*auc - 1

print("\nFinal Result(Training set) \n")
print(f"AUC Score:     {auc:.4f}")
print(f"Gini Score:    {gini_coef:.4f}")

#Scorecard parameters
PDO = 50  # points to double the odd
Base_Score = 600
Base_Odds = 5

factor = PDO/np.log(2)
offset = Base_Score - (factor * np.log(Base_Odds))

print(f"Scorecard Logic Used:")
print(f"Factor: {factor:.2f}")
print(f"Offset: {offset:.2f}")

#converts the probability of default into a credit score
def get_credit_score(probability):
    #Preventing Boundary conditions
    if probability == 0 : probability = 0.00001
    if probability == 1 : probability = 0.99999

    #considering probability as probability of default
    odds = (1-probability)/probability

    #using standard formula
    score = offset + (factor*np.log(odds))
    return int(np.clip(score,300,850))

#testing
y_probs = model.predict_proba(X_train_woe)[:,1]
scores = np.array([get_credit_score(p) for p in y_probs])

train_data['Credit_Score'] = scores
print("\n Sample Score \n")
print(train_data[['target','Credit_Score']].head(10))
print("\nAverage Score by status\n")
print(train_data.groupby('target')['Credit_Score'].mean())

plt.figure(figsize=(12, 7))
#good customer
sns.kdeplot(train_data[train_data['target'] == 0]['Credit_Score'], 
            color='green', label='Good Customers',fill = True,alpha=0.3)
#bad customer
sns.kdeplot(train_data[train_data['target'] == 1]['Credit_Score'], 
            color='red', label='Bad Customers', fill=True, alpha=0.3)
plt.title('Credit Score Distribution: Separation of Good vs Bad')
plt.xlabel('Credit Score')
plt.axvline(600, color='black', linestyle='--', label='Base Score')
plt.ylabel('Density')
plt.legend()
plt.grid(axis='y', alpha=0.3)
plt.show()

artifacts = {
    'model': model,                        
    'features': final_features,           
    'woe_map': woe_map,                      
    'binning_process': binning_process,     # saved bin edges
    'scorecard_params': {
        'offset': offset,
        'factor': factor
    },
    'cat_cols': categorical_col.tolist()
}
print("Saving model artifacts...")
with open('credit_score.pkl','wb') as f:
    pickle.dump(artifacts,f)
print("'credit_score.pkl' saved successfully!")