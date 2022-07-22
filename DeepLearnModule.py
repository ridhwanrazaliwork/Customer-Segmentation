import matplotlib.pyplot as plt
import scipy.stats as ss
import seaborn as sns
import pandas as pd
import numpy as np


from sklearn.metrics import confusion_matrix, classification_report,ConfusionMatrixDisplay
from sklearn.experimental import enable_iterative_imputer
from sklearn.linear_model import LogisticRegression
from sklearn.impute import IterativeImputer
from sklearn.impute import KNNImputer

def displot_graph(con_col,df):
    # continuous
    for i in con_col:
        plt.figure()
        sns.distplot(df[i])
        plt.show()

def countplot_graph(cat_col,df):
    # categorical
    for i in cat_col:
        plt.figure()
        sns.countplot(df[i])
        plt.show()

def groupby_plt(df,col1,col2):
    df.groupby([col1, col2]).agg({col2:'count'}).plot(kind='bar')

def confusion_mat(X,y,cat_col,selected_features,treshold=0.3):   
    def cramers_corrected_stat(confusion_matrix):
        """ 
        calculate Cramers V statistic for categorial-categorial association.
            uses correction from Bergsma and Wicher, 
            Journal of the Korean Statistical Society 42 (2013): 323-328
        """
        chi2 = ss.chi2_contingency(confusion_matrix)[0]
        n = confusion_matrix.sum()
        phi2 = chi2/n
        r,k = confusion_matrix.shape
        phi2corr = max(0, phi2 - ((k-1)*(r-1))/(n-1))    
        rcorr = r - ((r-1)**2)/(n-1)
        kcorr = k - ((k-1)**2)/(n-1)
        return np.sqrt(phi2corr / min( (kcorr-1), (rcorr-1)))

    for i in cat_col:
        print(i)
        matrix = pd.crosstab(X[i],y).to_numpy()
        print(cramers_corrected_stat(matrix))
        if cramers_corrected_stat(matrix) > treshold:
            selected_features.append(i)    



def Trad_FillNa_Con(df, con_col):
    for i in con_col:
        df[i] = df[i].fillna(df[i].median())

def Trad_FillNa_Cat(df, cat_col ):
    for i in cat_col:
        df[i] = df[i].fillna(df[i].mode()[0])

def KNNImpute(df,column_names):
    knn_im = KNNImputer()
    df = knn_im.fit_transform(df)
    df = pd.DataFrame(df)
    df.columns = column_names
    df.info()
    df.describe().T

def FloorDf(df, column):
    df[column] = np.floor(df[column])

def IterativeImpute(df):
    ii = IterativeImputer()
    df = ii.fit_transform(df)
    df = pd.DataFrame(df)
    df.columns = df.columns
    df.describe().T

def LogisticRegReverse(X,y,con_col,selected_features,treshold=0.8):
    for i in con_col:
        print(i)
        lr = LogisticRegression()
        lr.fit(np.expand_dims(y,axis=-1),X[i])
        print(lr.score(np.expand_dims(y,axis=-1),X[i]))
        if lr.score(np.expand_dims(y,axis=-1),X[i]) > treshold :
            selected_features.append(i)

def LogisticReg(X,y,con_col,selected_features,treshold=0.8):
    for i in con_col:
        print(i)
        lr = LogisticRegression(solver='liblinear')
        lr.fit(np.expand_dims(X[i],axis=-1), y)
        print(lr.score(np.expand_dims(X[i],axis=-1),y))
        if lr.score(np.expand_dims(X[i],axis=-1),y) > treshold :
            selected_features.append(i)

def ModelHist_plot(hist,plot1,plot2,leg1,leg2):
    plt.figure()
    plt.plot(hist.history[plot1])
    plt.plot(hist.history[plot2])
    plt.legend([leg1, leg2])
    plt.show()

def Model_Analysis(model,X_test,y_pred,y_test,lab):
    y_pred = model.predict(X_test)
    y_pred = np.argmax(y_pred,axis=1)
    y_true = np.argmax(y_test,axis=1)

    cm = confusion_matrix(y_true,y_pred)
    cr = classification_report(y_true,y_pred)

    labels =[lab]
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=labels)
    disp.plot(cmap=plt.cm.Blues)
    plt.show()

    print(cr)

def lineplot(df,column,start=100,stop=200):
    df_disp = df[start:stop]
    plt.figure()
    plt.plot(df_disp[column])
    plt.show()

def errorgraph(df,yerr1,yerr2,column,start=100,stop=200):
    df_disp = df[start:stop]
    yerr = df_disp[yerr1] - df_disp[yerr2]
    xaxis = np.arange(len(df_disp[column]))
    plt.figure()
    plt.errorbar(xaxis, df_disp[column], yerr =yerr)
    plt.show()

def Time_eval(y_test,predicted,xlab='Time',ylab='Data',leg=['Actual', 'Predicted']):
    plt.figure()
    plt.plot(y_test,color='red')
    plt.plot(predicted,color='blue')
    plt.xlabel(xlab)
    plt.ylabel(ylab)
    plt.legend(leg)
    plt.show()

def Time_eval_inverse(mms,y_test,predicted,xlab='Time',ylab='Data',leg=['Actual', 'Predicted']):
    actual_price = mms.inverse_transform(y_test)
    predicted_price = mms.inverse_transform(predicted)
    plt.figure()
    plt.plot(y_test,color='red')
    plt.plot(predicted,color='blue')
    plt.xlabel(xlab)
    plt.ylabel(ylab)
    plt.legend(leg)
    plt.show() 

def boxplot(df,con_col,nrows=1,ncols=1, size1=(30,40)):
    fig, ax = plt.subplots(nrows, ncols, figsize=size1)
    df[con_col].plot.box(layout=(nrows, ncols), 
                subplots=True, 
                ax=ax, 
                vert=False, 
                sharex=False)
    plt.tight_layout()
    plt.show()