# -*- coding: utf-8 -*-
"""
Created: ouxiaogu, 2018-10-31 11:11:40

The class for contour selection sklearn classifiers
Keep sklearn models as clear as possible, don't need any MXP data structure

Last Modified by:  ouxiaogu
"""
import numpy as np
import pandas as pd
from sklearn import svm, tree, ensemble
# from sklearn.metrics import confusion_matrix

from ContourSelBaseModel import ContourSelBaseModel

import sys
import os.path

sys.path.insert(0, (os.path.dirname(os.path.abspath(__file__)))+"/../../../libs/common/")
from logger import logger
log = logger.getLogger(__name__)

class ContourSelClfModel(ContourSelBaseModel):
    '''
    classification model type here includes {'SVC': 'SVM', 'DT': 'Decision Tree', 'RF': 'Random Forest'}
    '''
    modeltype = 'clf'
    
    def __init__(self, useNeighborFeatures=True, **kwargs):
        self.useNeighborFeatures = True

    @staticmethod
    def computeFeatureScalar(X_cal):
        Xmin = X_cal.min(axis=0)
        Xmax = X_cal.max(axis=0)
        Xminmax = pd.DataFrame(data= np.array([Xmin, Xmax]), columns=X_cal.columns, index=['min', 'max'])
        log.info("feature scaling into range 0~1, original range:\n{}".format(Xminmax))
        return Xminmax
    
    @staticmethod
    def applyFeatureScalar(X, Xminmax):
        Xmin = Xminmax.loc['min',:].values
        Xmax = Xminmax.loc['max',:].values
        scaling = lambda X_Arr: np.array([(X_Arr[i] - Xmin[i])/(Xmax[i] - Xmin[i]) for i in range(len(Xminmax.columns)) ])
        X.loc[:, Xminmax.columns] = X.loc[:, Xminmax.columns].apply(scaling, axis=1)
        return X

    def calibrate(self, X_cal, y_cal, X_ver, y_ver, model_type='DT'):
        # feature scaling
        Xminmax = ContourSelClfModel.computeFeatureScalar(X_cal)
        X_cal = ContourSelClfModel.applyFeatureScalar(X_cal, Xminmax)
        X_ver = ContourSelClfModel.applyFeatureScalar(X_ver, Xminmax)

        # calibrate classification model
        if model_type == 'SVM':
            model = ContourSelClfModel.calSVCModel(X_cal, y_cal)
        elif model_type == 'RF':
            model = ContourSelClfModel.calRFModel(X_cal, y_cal)
        else:
            model = ContourSelClfModel.calDTModel(X_cal, y_cal)

        # calibration performance
        _, cm_cal = ContourSelClfModel.predict(model, X_cal, y_cal, 'CAL')
        _, cm_ver = ContourSelClfModel.predict(model, X_ver, y_ver, 'VER')
        self.printModelPerformance(cm_cal, usage='CAL')
        self.printModelPerformance(cm_ver, usage='VER')
        return model, Xminmax, cm_cal, cm_ver

    @staticmethod
    def calSVCModel(X_cal, y_cal):
        ContourSelClfModel.modeltype = 'SVM'
        clf = svm.SVC(kernel='linear', class_weight='balanced', random_state=0) # {0: 10, 1: 1}
        model = clf.fit(X_cal, y_cal)

        log.debug("SVC model parameter setting:\n{}".format(model.get_params()))
        modelform = pd.DataFrame(data=clf.coef_.flatten(), index=X_cal.columns)
        modelform.loc['intercept', 0] = clf.intercept_
        log.info("SVC model form:\n{}".format(modelform))
        return model

    @staticmethod
    def calDTModel(X_cal, y_cal):
        ContourSelClfModel.modeltype = 'DT'
        clf = tree.DecisionTreeClassifier(random_state=0, max_depth=len(X_cal.columns)+1, min_samples_split=3)
        model = clf.fit(X_cal, y_cal)

        log.debug("Decision Tree model parameter setting:\n{}".format(model.get_params()))
        df_feature_importance = pd.DataFrame(data=model.feature_importances_.flatten(), index=X_cal.columns)
        log.info("Decision Tree model feature importance:\n{}".format(df_feature_importance))
        return model

    @staticmethod
    def calRFModel(X_cal, y_cal):
        ContourSelClfModel.modeltype = 'RF'
        clf = ensemble.RandomForestClassifier(n_estimators=10, random_state=0)
        model = clf.fit(X_cal, y_cal)
        log.debug("Random Forest model parameter setting:\n{}".format(model.get_params()))
        df_feature_importance = pd.DataFrame(data=model.feature_importances_.flatten(), index=X_cal.columns)
        log.info("Random Forest model feature importance:\n{}".format(df_feature_importance))

    @staticmethod
    def predict(model, X, y_true=None, usage='CAL'):
        y_pred = model.predict(X)

        cm = np.zeros((2,2), dtype=int)
        if y_true is not None:
            df = pd.DataFrame(data= np.array([y_true.values, y_pred]).T, 
                columns=[ContourSelBaseModel.tgtColName, ContourSelBaseModel.outColName])
            cm = ContourSelBaseModel.computeConfusionMatrix(df)
        return y_pred, cm