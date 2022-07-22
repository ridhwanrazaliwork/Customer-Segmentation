#%% Import library
import os
import pickle
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import datetime

from tensorflow.keras.utils import plot_model
from tensorflow.keras.callbacks import ModelCheckpoint,TensorBoard, EarlyStopping
from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report
from sklearn.impute import KNNImputer
from sklearn.preprocessing import MinMaxScaler,LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras import Sequential, Input
from DeepLearnModule import displot_graph,countplot_graph,boxplot,confusion_mat
from DeepLearnModule import LogisticReg,ModelHist_plot
#%% Data loading & Path
#Path
CSV_PATH = os.path.join(os.getcwd(),'Data','Train.csv')
MODEL_SAVE_PATH = os.path.join(os.getcwd(),'Models', 'model.h5')
OHE_PATH = os.path.join(os.getcwd(),'Models','ohe.pkl')
BEST_MODEL_PATH = os.path.join(os.getcwd(),'Models','best_model.h5')
LOGS_PATH = os.path.join(os.getcwd(),'Logs',datetime.datetime.now().
                    strftime('%Y%m%d-%H%M%S'))


# Data loading
df = pd.read_csv(CSV_PATH)

# Checking data
df
df.info()
df.describe().T
# for communication_type and prev_campaign "unknown" label can be NaNs

# Drop id column, not needed for this model
df = df.drop(labels='id' , axis=1)
#%% constant
cat_col = list(df.columns[df.dtypes=='object'])
cat_col.append('term_deposit_subscribed')

con_col = list(df.columns[(df.dtypes=='int64') | (df.dtypes=='float64')])
con_col.remove('term_deposit_subscribed')
#%% Data visualization
displot_graph(con_col,df)
# A lot of outliers in balance, last_contact_duration, num_contacts_in_campaign
# outlier in num_contacts_prev_campaign

countplot_graph(cat_col,df)

# display unique value percentages for target
item_counts = df["term_deposit_subscribed"].value_counts(normalize=True)
print(item_counts)
# imbalance datasets, more people not subscribe 89% than people subscribe 11%

# box plot (to see outliers more clearly)
boxplot(df=df,con_col=con_col,nrows=7,ncols=1,size1=(30,40))

# %%  Data cleaning
# check NaNs percentage
df.isna().sum()/len(df)*100

# copy df to compare after impute
df_copy = df

# drop days_since_prev_campaign_contact because got 82% NaNs value
df = df.drop(labels=['days_since_prev_campaign_contact'],axis=1)

# Label encoder to change object to int/float in cat_col
for i in cat_col:
    if i == 'term_deposit_subscribed':
        continue
    else:
        le = LabelEncoder()
        temp = df[i]
        temp[temp.notnull()] = le.fit_transform(temp[df[i].notnull()]) # label encoder without touching NaNs
        df[i] = pd.to_numeric(df[i],errors='coerce')
        PICKLE_SAVE_PATH = os.path.join(os.getcwd(),'Models',i+'_encoder.pkl')
        with open(PICKLE_SAVE_PATH,'wb') as file:
            pickle.dump(le,file)

# KNN impute
columns_name = df.columns
knn_im = KNNImputer()
df = knn_im.fit_transform(df)
df = pd.DataFrame(df)
df.columns = columns_name

# update list after drop
# con_col = list(con_col & df.columns)
con_col.remove('days_since_prev_campaign_contact')

# check dataframe
df.isna().sum()
df.describe().T
# no NaNs

# check df after impute
df_copy.describe().T - df.describe().T
# 'ok' no big changes after impute lesser than 5%

# change all cat_col to int after impute
for i in cat_col:
    df[i] = np.floor(df[i]).astype(int)

# check
df.info()
#%% Features selection
# drop target from cat_col
cat_col.remove('term_deposit_subscribed')

# Define X,y
X = df.drop(labels='term_deposit_subscribed',axis=1)
y = df['term_deposit_subscribed']

# cramers V for cat vs cat d ata
selected_features = []
confusion_mat(X,y,cat_col,selected_features,treshold=0.3)
# very low correlation from cat_col lower than 50%

# logistic regression for cat vs con data
LogisticReg(X,y,con_col,selected_features,treshold=0.8)
# good correlation from con_col most higher than 80%

# %% Data preprocessing
df.info()
# Takes selected_features from above
# Append missing target
selected_features.append('term_deposit_subscribed')

df = df.loc[:,selected_features]
X = df.drop(labels='term_deposit_subscribed',axis=1)
y= df['term_deposit_subscribed'].astype(int)
 
# MinMaxScaler
mms = MinMaxScaler()
X= mms.fit_transform(X)

MMS_PATH = os.path.join(os.getcwd(), 'Models', 'mms.pkl')


# OHE
ohe = OneHotEncoder(sparse=False)
y = ohe.fit_transform(np.expand_dims(y,axis=-1))

X_train, X_test, y_train, y_test = train_test_split(X,y,
                                    test_size=0.3,random_state=123)

#%% Deep learning Model

nb_class = len(np.unique(y,axis=0))
model = Sequential()
model.add(Input(shape=np.shape(X_train)[1:]))
model.add(Dense(128,activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.3))
model.add(Dense(128,activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.3))
model.add(Dense(128,activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.3))
model.add(Dense(nb_class,activation='softmax'))
model.summary()

# plot model
plot_model(model, show_shapes=True, show_layer_names=(True))
# %% Model compile
model.compile(optimizer='adam',loss='categorical_crossentropy',
                metrics=['acc'])

# Added model checkpoint
mdc = ModelCheckpoint(BEST_MODEL_PATH,monitor='val_acc',
                        save_best_only=True,mode='max')

                    
tensorboard_callback = TensorBoard(log_dir=LOGS_PATH,histogram_freq=1)

# Early stopping
early_callback = EarlyStopping(monitor='val_loss',patience=5)                       

hist = model.fit(X_train,y_train,epochs=100,verbose=1,
                    validation_data=(X_test,y_test),
                    callbacks=[mdc,tensorboard_callback, early_callback])

#%% plot performance graph

# Check hist keys
print(hist.history.keys())

ModelHist_plot(hist,'loss','val_loss','training_loss','val_loss')

ModelHist_plot(hist,'acc','val_acc','training_acc','val_acc')


y_pred = np.argmax(model.predict(X_test),axis=1)
y_test = np.argmax(y_test,axis=1)

print(classification_report(y_test,y_pred))

#%% Save all model and scaler
# save and load best model
best_model = load_model(BEST_MODEL_PATH)
best_model.summary()

#check best model
y_pred = np.argmax(best_model.predict(X_test),axis=1)
y_test = y_test

print(classification_report(y_test,y_pred))
print(confusion_matrix(y_test,y_pred))

# save OHE,MMS,Model
with open(OHE_PATH,'wb') as file:
    pickle.dump(OHE_PATH,file)

with open(MMS_PATH,'wb') as file:
    pickle.dump(mms,file)

model.save(MODEL_SAVE_PATH)

# %% Confusion matrix plot
fig, ax = plt.subplots(1, 1, figsize=(7, 7))
ConfusionMatrixDisplay.from_predictions(y_test, y_pred, normalize='all', ax=ax, display_labels=['not subscribed', 'subscribed'])
plt.tight_layout()
plt.show()