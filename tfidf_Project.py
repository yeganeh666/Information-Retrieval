import sys,os
import csv
import tkinter as tk
from tkinter import ttk

import matplotlib.pyplot as plt
#import csv
import sklearn
import pickle

import pandas as pd
import numpy as np
import nltk
import seaborn as sns
import os
#import xlwings as xw
import warnings
import string
from decimal import Decimal
#from wordcloud import WordCloud
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer,TfidfVectorizer
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV,train_test_split,StratifiedKFold,cross_val_score,learning_curve
from collections import Counter
from sklearn.preprocessing import StandardScaler
from pandastable import Table, TableModel
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.decomposition import PCA
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from openpyxl import load_workbook
from tkinter import *
from tkinter import ttk
from PIL import Image, ImageTk
from sklearn.metrics import multilabel_confusion_matrix
from tkinter import messagebox as msg
from sklearn.metrics import classification_report

 
class Window(Frame):
    def __init__(self, master=None):
        Frame.__init__(self, master)
        self.master = master
           
        menu = Menu(self.master)
        self.master.config(menu=menu)
       # blankmenu = Menu(menu)
       # menu.add_cascade(label="".ljust(130), menu=blankmenu)
        fileMenu = Menu(menu)
        fileMenu.add_command(label="feature to tkinter" ,command=self.print_f_tk)
        fileMenu.add_command(label="feature to excel" ,command=self.f_excel)
        fileMenu.add_command(label="feature vectors to tkinter" ,command=self.fv_tk)
        fileMenu.add_command(label="feature vectors to excel" ,command=self.fv_excel)
          
        menu.add_cascade(label="tf_idf", menu=fileMenu)

        
       
        
        GanMenu = Menu(menu)
        menu.add_cascade(label="split dataset", menu=GanMenu)
        GanMenu.add_cascade(label="train", command=self.get_train)
        GanMenu.add_cascade(label="test", command=self.get_test)
        GanMenu.add_command(label="validate",command=self.get_validate) 
        GanMenu.add_cascade(label="split dataset dimension", command=self.table_fv)
      
        CalcMenu = Menu(menu)
        menu.add_cascade(label="calculation", menu=CalcMenu)
        CalcMenu.add_cascade(label="svc_knn_nb_dt", command=self.print_predict)
        CalcMenu.add_cascade(label="Prec_Recall_F1 Score_Accuracy ", command=self.p_f_r_a1)
        CalcMenu.add_cascade(label="Prec_Recall_F1 Score_Accuracy each class ", command=self.p_f_r_a2)
        conf = Menu(menu)
        menu.add_cascade(label="confusion matrix", menu=conf)
        conf.add_cascade(label="plot", command=self.plot_matrix)
        #conf.add_cascade(label="confusion matrix", command=self.print_matrix)
        PridectMenu = Menu(menu)
        menu.add_cascade(label="Predection", menu=PridectMenu,command=self.pr_predict)
        PridectMenu.add_cascade(label="Predection",command=self.pr_predict)
        self.pack(fill=BOTH, expand=1)
    def pri(self,txt,my_w):
      txt=txt.get()
      txt=[txt]
      mnb = MultinomialNB(alpha=0.2)
      prd = vec.transform(txt)
      mnb.fit(X_train,y_train)
      x = mnb.predict(prd)
      if x == 0:
         msg.showinfo(title="classification news", message="news is  Social.")
      elif x == 1:   
         msg.showinfo(title="classification news", message="news is  Economic.")
      elif x == 2:
          msg.showinfo(title="classification news", message="news is  international.")
      elif x == 3:
          msg.showinfo(title="classification news", message="news is  Political.")
      elif x == 4:
          msg.showinfo(title="classification news", message="news is  Science.")
      elif x == 5:
          msg.showinfo(title="classification news", message="news is  Cultural.")
      elif x == 6:
          msg.showinfo(title="classification news", message="news is  Sports.")
      elif x == 7:
          msg.showinfo(title="classification news", message="news is  Medical.")
      
       
      
        
    def pr_predict(self):
        my_w = tk.Toplevel(root)
        my_w.geometry("600x400")
        lable1 = Label(my_w, text="please enter your news")
        lable1.place(x=50, y=10)
        txt = Entry(my_w,width=150, justify="right")
        txt.place(x=180, y=10)
        btn = Button(my_w, text ="prediction", fg ="red",command=lambda:self.pri(txt,my_w)).place(x=285,y=140)
        
        
    
        
    def return_spli(self,x1,y1):
        df2 = pd.DataFrame(x1.toarray())
        rdf=df2.shape[0]
        cdf=df2.shape[1]
        df3=pd.DataFrame(y1)
        df4=pd.concat([df2.reset_index(drop=1).add_suffix('_1'),df3.reset_index(drop=1).add_suffix('_2')], axis=1).fillna('')
        return df4
    def print_df(self,df4):
        frame = Toplevel(self.master)
        table=Table(frame)
      
        dftable=Table(frame,dataframe=df4,showtoolsbar=True,showstatusbar=True, enable_menus=True,x=120,y=150)
        dftable.show()
    def get_train(self):
        
       
        df4=self.return_spli(X_train,y_train)
        self.print_df(df4)
    
  
       
       
    def get_validate(self):
        df4=self.return_spli(X_val,y_val)
        self.print_df(df4)
    
  
       
           
       
    def get_test(self):
        df4=self.return_spli(X_test,y_test)
        self.print_df(df4)
    
  
       
        
        
        
    def table_fv(self):
        df1 = pd.DataFrame(X_train.toarray())
        rdf1=df1.shape[0]
        cdf1=df1.shape[1]
        df2 = pd.DataFrame(y_train)
        rdf2=df2.shape[0]
        cdf2=df2.shape[1]
        df3 = pd.DataFrame(X_test.toarray())
        rdf3=df3.shape[0]
        cdf3=df3.shape[1]
        df4 = pd.DataFrame(y_test)
        rdf4=df4.shape[0]
        cdf4=df4.shape[1]
        df5 = pd.DataFrame(X_val.toarray())
        rdf5=df5.shape[0]
        cdf5=df5.shape[1]
        df6 = pd.DataFrame(y_val)
        rdf6=df6.shape[0]
        cdf6=df6.shape[1]
        pr_dict={'x_train':{'row':rdf1,'col:': cdf1},'y_train':{'row':rdf2,'col:': cdf2},'x_test':{'row':rdf3,'col:': cdf3},'y_test':{'row':rdf4,'col:': cdf4},'x_validate':{'row':rdf5,'col:': cdf5},'y_validate':{'row':rdf6,'col:': cdf6}}
        self.print_df(pr_dict.items())
    def p_f_r_a1(self):   
        frame = Toplevel(self.master)
        svc=self.retu_train_test()
        svc.fit(X_train, y_train)
        
        y_pred = svc.predict(X_test)
        
        pr=precision_score(y_test, y_pred,average='weighted')
        re=recall_score(y_test, y_pred,average='weighted')
        fs=f1_score(y_test, y_pred,average='weighted')
        ac=accuracy_score(y_test, y_pred)
        pr_dict={'Precision':pr,'Recall:': re,'F1 Score:': fs,'Accuracy:': ac}
        
        self.print_df(pr_dict.items())
    #    dftable = Table(frame, dataframe=dft)
     #   dftable.textcolor = 'blue'
        
        
    #    dftable.show()
    def p_f_r_a2(self):
        frame = Toplevel(self.master)
        svc=self.retu_train_test()
        svc.fit(X_train, y_train)
        
        y_pred = svc.predict(X_test)
        
     #   pr=precision_score(y_test, y_pred,average='weighted')
      #  re=recall_score(y_test, y_pred,average='weighted')
      #  fs=f1_score(y_test, y_pred,average='weighted')
     #   ac=accuracy_score(y_test, y_pred)
      #  pr_dict={'Precision':pr,'Recall:': re,'F1 Score:': fs,'Accuracy:': ac}
        
      #  self.print_df(pr_dict.items())
        rpd = classification_report(y_test, y_pred, output_dict=True)
        dft=pd.DataFrame.from_dict(rpd)
        dft=(dft).T
        dft = dft.reset_index()
        #dft.columns = ['precision', 'recall', ' f1-score', 'support',' accuracy','macro avg','weighted avg']
        dftable = Table(frame, dataframe=dft)
        dftable.textcolor = 'blue'
        
        
        dftable.show()
    
        
        
        #print(classification_report(y_test, y_pred))
        
    
    def retu_train_test(self):
        sc = StandardScaler(with_mean=False)
        sc.fit(X_train)
        X_train_std = sc.transform(X_train)
        X_test_std = sc.transform(X_test)
        svc = SVC(kernel='linear', C=10.0, random_state=1)
        svc.fit(X_train, y_train)
#
# Get the predictions
#
        
        return svc
    def plot_matrix(self):
        svc=self.retu_train_test()
        
        svc.fit(X_train, y_train)

        y_pred = svc.predict(X_test)

        conf_matrix = confusion_matrix(y_true=y_test, y_pred=y_pred)

        fig, ax = plt.subplots(figsize=(5, 5))
        ax.matshow(conf_matrix, cmap=plt.cm.Oranges, alpha=0.3)
        for i in range(conf_matrix.shape[0]):
          for j in range(conf_matrix.shape[1]):
              ax.text(x=j,y=i,s=conf_matrix[i, j],va='center',ha='center',size='xx-large') 
        plt.xlabel('Predictions', fontsize=18)
        plt.ylabel('Actuals', fontsize=18)
        plt.title('Confusion Matrix', fontsize=18)
        #print(classification_report(y_test, y_pred))
        plt.show()
        
    
        
    def print_df(self,prid_vector):
        frame = Toplevel(self.master)
        
        #df = pd.DataFrame(dct)
        dft = pd.DataFrame.from_dict(prid_vector)
        dftable = Table(frame, dataframe=dft)
        dftable.textcolor = 'blue'
        
        #dftable.set_title("Top 10 Fields of Research by Aggregated Funding Amount")
        dftable.show()
    def train(self,acu_dict, features, targets):
        acu_dict.fit(features, targets)    
    def print_predict(self):
        features = vector
        svc = SVC(kernel='sigmoid', gamma=1.0)
        knn = KNeighborsClassifier(n_neighbors=49)
        mnb = MultinomialNB(alpha=0.2)
        dtc = DecisionTreeClassifier(min_samples_split=7, random_state=111)
        acu_dict = {'SVC' : svc,'KN' : knn, 'NB': mnb, 'DT': dtc}
        prid_vector = []
        for k,v in acu_dict.items():
           self.train(v, X_train, y_train)
           pd = self.predict(v, X_test)
           prid_vector.append((k, [accuracy_score(y_test , pd)]))
        self.print_df(prid_vector)
        
    def predict(self,acu_dict, features):
        return (acu_dict.predict(features))       
    def print_f_tk(self):
        frame = Toplevel(self.master)
        
        #df = pd.DataFrame(dct)
        ts = pd.DataFrame.from_dict(feu)
        dftable = Table(frame, dataframe=ts)
        
        dftable.show()
    def f_excel(self):    
        df = pd.DataFrame(feu)
        df.to_excel(r'D:\my\news1.xlsx')
    def fv_excel(self):    
        df1 = self.return_fv()
        
        df1.to_excel(r'D:\my\news2.xlsx')
       # df.to_excel(r'D:\my\news1.xlsx')
    def pr(self):
        df1=self.return_fv()
        frame = Toplevel(self.master)
        
        dftable = Table(frame, dataframe=df1)
        dftable.show()
        #frame = Toplevel(self.master)
        #dftable = Table(frame, dataframe=return_fv(vector,vec))
        #dftable.show()
        #print(self.return_fv())
    def return_fv(self):
        features = vector
        name = vec.get_feature_names()
        df4=pd.DataFrame(features.toarray(), columns=name)
        return df4
    def return_fv1(self):
        features = vector
        df2 = pd.DataFrame(features.toarray())
        name = vec.get_feature_names()
        df3=pd.DataFrame(name)
        df4 = pd.concat([df2.reset_index(), df3], axis=1)
        return df4
    def fv_tk(self):
        df1=self.return_fv()
        frame = Toplevel(self.master)
        
        dftable = Table(frame, dataframe=df1)
        dftable.show()
def delet_stopword(text):
    #df = pd.read_excel(r'D:\my\spam2.xlsx', sheet_name='Sheet1') # can also index sheet by name or fetch all sheets
    #mylist = df['column1'].tolist()
    mylist = ["در", "این",":","از","با","به","؛",":","،","آقا","آقای","آقایان","آمد","آمدن","core","acd","anc","led","آن","آنان","آنها","آنچه","آنکه","max","pro","p20","the","xs","آب"]

    text = text.translate(str.maketrans('', '', string.punctuation))
    text = [word for word in text.split() if word not in  mylist]
    
    text = [word for word in text if word not in stopwords.words('english')]

    return " ".join(text)

root = Tk()
app = Window(root)
root.wm_title("tf idf projects")

root.geometry("550x300+300+150")
root.resizable(width=True, height=True)
data = pd.read_excel(r'D:\my\dev1.xlsx', sheet_name='Sheet1')
warnings.filterwarnings('ignore')
pd_options = {
#'display.max_rows'    : 500,
#'display.max_columns' : 500,
'display.width'       : 1000,
'display.precision'   :4,
}

[pd.set_option(option, setting) for option, setting in pd_options.items()]
data['text'] = data['text'].apply(delet_stopword)
counter = CountVectorizer(binary=True)

f_text = counter.fit_transform(data['text'])
#feu = {k:[v] for k,v in list(counter.vocabulary_.items())[:200]}
feu = {k:[v] for k,v in list(counter.vocabulary_.items())}
vec = TfidfVectorizer(min_df=5, max_df=0.5)
vector = vec.fit_transform(data['text'].astype('U'))
X_train, X_test, y_train, y_test = train_test_split(vector, data['label_id'], test_size=0.2, random_state=111)

X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=1)
root.mainloop()

