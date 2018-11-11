#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 10 09:55:53 2018

@author: huiyuzhang
"""
import pandas as pd
import os
import re
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from lastest_class import classifier_workshop
from sklearn import preprocessing
from sklearn.feature_extraction.text import TfidfVectorizer


the_path='/Users/huiyuzhang/Desktop/workshop_data'
the_framework=classifier_workshop()
pd_data=the_framework.walk_data(the_path)
pd_data=the_framework.clean_data(pd_data)


################## Random Forest #########################
model_in=RandomForestClassifier(n_estimators=100,max_depth=2,random_state=0)
the_model_out, the_vec_out, the_label_enc_out=the_framework.train(pd_data,model_in)
thresh=0.05
my_text='fishing'
my_pred_value, checker=the_framework.classifier(the_model_out, the_vec_out,the_label_enc_out,thresh, my_text)

######################################
model_in=GradientBoostingClassifier(n_estimators=100,max_depth=2,random_state=0)
the_model_out, the_vec_out, the_label_enc_out=the_framework.train(pd_data,model_in)
thresh=0.05
my_text='fishing'
my_pred_value, checker=the_framework.classifier(the_model_out, the_vec_out,the_label_enc_out,thresh, my_text)
##########################

#the_pred_value=the_framework.classifer(the_model_out)
#vectorizer=TfidfVectorizer(max_features=1000,ngram_range=(1,3))

#tdm=pd.DataFrame(vectorizer.fit_transform(pd_data.body).toarray())
#tdm.columns=vectorizer.get_feature_names()
#label_enc=preprocessing.LabelEncoder()
#the_labels=label_enc.fit_transform(pd_data.label)

#model.fit(tdm,the_labels)

#the_sample=vectorizer.transform(['trail mix is good to eat while I climb'])
#the_val=model.predict(the_sample)
#label_enc.inverse_transform(the_val)

#the_val=model.predict([tdm.loc[258]])
#label_enc.inverse_transform(the_val)
#tdm.loc[1]


#the_prob=model.predict_proba(the_sample)
#the_raw=model.classes_
#the_pred=pd.DataFrame({'prob':the_prob[0],'labels':the_raw})
#the_pred=pd.DataFrame({'prob':the_prob[0],'labels':label_enc.inverse_transform(the_raw)})


#thresh=0.5
#import numpy as np
#the_max_val=np.max(the_pred.prob)
#if(the_max_val >=thresh):
#    the_confident_tmp=the_pred.prob==the_max_val
#    the_predicted_value=the_pred[the_confident_tmp]
    

#the_confident_tmp=the_pred.prob>=thresh
#the_predicted_value=the_pred[the_confident_tmp]


    


#print(pd_data)

    

