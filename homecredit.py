import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import roc_auc_score
import warnings
warnings.simplefilter(action = 'ignore', 
                      category = FutureWarning)
import seaborn as sb
from matplotlib import pyplot as plt
from tqdm import trange, tqdm_notebook

class HomeCreditDataTable:
    
    """
    This is a basic class for dealing with initial tables with LGBMClassifier
    """
    
    def __init__(self):
        self._data = None
        self._target = None
        self._train = None
        self._test = None
    
    @property
    def data(self):
        return self._data.copy()
    @data.setter
    def data(self, df):        
        self._data = df 
        if self._train is not None:
            self._train = list(set(self._train).intersection(set(self._data.index.tolist())))
            self._target = self._target[self._train]
        else:
            self._train = self._data.index.tolist()     
    @data.deleter 
    def data(self):        
        self._data = None
        self._train = None
        self._target = None 
        self._test = None
    
    @property
    def train(self):
        return self._data.join(self._target).loc[self._train, :].copy()
    @property
    def target(self):
        return self._target.copy()
    @train.setter
    def train(self, df):    
        if self._data is None:
            self._data = df.drop('TARGET', axis = 1)      
            self._target = df.TARGET
            self._train = df.index.tolist()
        else:
            raise ValueError('TrainData already exists, check what to use')    
    @train.deleter
    def train(self):
        self._train = None
    @target.deleter
    def target(self):
        self._target = None
        
    @property
    def test(self):
        return self._data.loc[self._test, :].copy()
    @test.setter
    def test(self, df):    
        if self._data is None:
            raise ValueError('TrainData is absent')
        else:
            self._test = df.index.tolist()
            self._data = pd.concat((self._data, df))             
    @test.deleter
    def test(self):    
        self._test = None
    
    def set_column(self, column, col_name = None):
        if col_name is None:
            self._data[column.name] = column
        else:
            self._data[col_name] = column.rename(col_name)
    
    @property
    def n_splits(self):
        return self._n_splits
    @property
    def train_cv(self):
        return self._train_cv
    @property
    def valid_cv(self):
        return self._valid_cv
    def cv_split(self, stratified = True, n_splits = 5, 
                 shuffle = True, random_state = 0):       
        self._n_splits = n_splits
        self._train_cv = []
        self._valid_cv = []        
        if not isinstance(self._data, pd.DataFrame):
            raise TypeError('Invalid DataTable. A DataFrame should be passed')          
        if stratified:            
            cv = StratifiedKFold(n_splits = n_splits, 
                                 shuffle = shuffle, 
                                 random_state = random_state)
        else:
            cv = KFold(n_splits = n_splits, 
                       shuffle = shuffle,
                       random_state = random_state)                
        for train_idx, valid_idx in cv.split(self._data.loc[self._train],
                                             self._target):
            self._train_cv.append(self._data.index[train_idx].tolist())
            self._valid_cv.append(self._data.index[valid_idx].tolist())
    
    @staticmethod
    def _woe_iv(column, target):
            data = pd.merge(left = column.to_frame('c'), 
                            right = target.to_frame('t'), 
                            left_index = True,
                            right_index = True)
            cross_tab = pd.crosstab(data.c, data.t)
            cats = cross_tab.index.tolist()
            distr_goods = (cross_tab[1].values + 1)/(cross_tab[1].sum() + 1)
            distr_bads = (cross_tab[0].values + 1)/(cross_tab[0].sum() + 1)
            woe = np.log(distr_goods / distr_bads)
            iv = np.sum((distr_goods - distr_bads) * woe)            
            return pd.DataFrame(data = woe, 
                                index = cats, 
                                columns = [column.name + '_woe']), iv    
    def count_woe_iv(self, column_names):        
        if isinstance(column_names, str):
            column_names = [column_names]
        for col in column_names:
            column = self._data[col][self._train]
            target = self._target
            woe = self._woe_iv(column, target)
            try:
                self.woe[col] = woe[0]
                self.iv[col] = woe[1]
            except AttributeError:
                self.woe = {}
                self.woe[col] = woe[0]
                self.iv = {}
                self.iv[col] = woe[1]                
    def woe_encode(self, column_names = None):  
        if column_names is None:
            column_names = self.data.select_dtypes(object).columns.tolist()
        if isinstance(column_names, str):
            column_names = [column_names]        
        encoders = []
        train_encoded = []
        valid_encoded = []
        test_encoded = []        
        for split in range(self._n_splits):
            tr_s_e = pd.DataFrame(index = self._train_cv[split])
            va_s_e = pd.DataFrame(index = self._valid_cv[split])
            te_s_e = pd.DataFrame(index = self._test)
            enc_s = {}            
            for col in column_names:
                column = self._data[col][tr_s_e.index]
                target = self._target[tr_s_e.index]
                encoder = self._woe_iv(column, target)[0].iloc[:, 0]
                enc_s[col] = encoder
                tr_s_e[col] = self._data[col][tr_s_e.index].map(encoder)
                va_s_e[col] = self._data[col][va_s_e.index].map(encoder)
                te_s_e[col] = self._data[col][te_s_e.index].map(encoder)
            encoders.append(enc_s)
            train_encoded.append(tr_s_e)
            valid_encoded.append(va_s_e)
            test_encoded.append(te_s_e) 
        self._encoded_cols = (train_encoded,
                              valid_encoded,
                              test_encoded,
                              encoders)
    @property
    def encoded_cols(self):
        return self._encoded_cols
    
    def binning(self, column_names, zeros_as_bin = True, nbins = None):
        if isinstance(column_names, str):
            column_names = [column_names] 
        for column in column_names:
            transformed = self._data[[column]]
            if zeros_as_bin:
                zeros = transformed[transformed[column] == 0]
                transformed = transformed.replace({0: np.nan})
            totransform = transformed.dropna().copy()
            if nbins == None:
                n = totransform.shape[0]
                nbins = int(n ** (1/3))
            totransform[column + '_bins'] = pd.cut(totransform[column], 
                                                   bins = nbins, 
                                                   labels = False)
            totransform[column + '_bins'] += 1
            transformed[column + '_bins'] = transformed.join(totransform[[column + '_bins']])[column + '_bins']
            if zeros_as_bin:
                transformed.loc[zeros.index, column + '_bins'] = 0
            transformed = transformed.fillna(-1)         
            self._data[column + '_bins'] = transformed[column + '_bins']
    
    @property
    def model(self):
        return self._model
    @model.setter
    def model(self, model):
        self._model = model
    @model.deleter
    def model(self):
        self._model = None
    def _predict_for_split(self, split, predictors, 
                           predict = False,
                           metric = roc_auc_score): 
        if isinstance(predictors, str):
            predictors = [predictors]
        x_train = self._data.loc[self._train_cv[split], predictors]
        y_train = self._target[self._train_cv[split]]
        x_valid = self._data.loc[self._valid_cv[split], predictors]
        y_valid = self._target[self._valid_cv[split]]
        if predict:
            x_test = self._data.loc[self._test, predictors]
        try:
            enc_cols = self._encoded_cols[0][split].columns.tolist()
            enc_cols = [c for c in enc_cols if c in predictors]
        except AttributeError:
            enc_cols = []
        if len(enc_cols):
            x_train[enc_cols] = self._encoded_cols[0][split][enc_cols]
            x_valid[enc_cols] = self._encoded_cols[1][split][enc_cols]           
            if predict: 
                x_test[enc_cols] = self._encoded_cols[1][split][enc_cols]        
        try:
            self._model.fit(x_train, y_train,
                    eval_set=[(x_valid, y_valid)],
                    eval_metric = 'auc', 
                    early_stopping_rounds = self.early_stop_rounds,
                    verbose = False) 
        except:
            self._model.fit(x_train, y_train)
        try:
            val_preds = self._model.predict_proba(x_valid,
                                              num_iterations = self._model.best_iteration_)[:, 1]
        except:
            val_preds = self._model.predict_proba(x_valid)[:, 1]        
        score = metric(y_valid, val_preds)
        if predict:
            result = [score]
            result.append(val_preds)
            try:
                te_preds = self._model.predict_proba(x_test,
                                                 num_iterations = self._model.best_iteration_)[:, 1]
            except:
                te_preds = self._model.predict_proba(x_test)[:, 1]
            result.append(te_preds)
        else:
            result = score
        return result

    @property
    def predictors_scores(self):
        return self._predictors_scores.copy()    
    def evaluate_predictors(self, predictors = None, metric = roc_auc_score):
        if predictors is None:
            predictors = self._data.columns.tolist()  
        scores = pd.DataFrame(index = predictors,
                              columns = ['cv_score'],
                              data = np.zeros((len(predictors), 1)))  
        for split in range(self._n_splits):
            for p in predictors:
                try:
                    score = self._predict_for_split(split = split, 
                                                    predictors = p, 
                                                    metric = metric)
                except:
                    print(p + ': something went wrong')
                    score = 0
                finally:
                    scores.loc[p, 'cv_score'] += score / self._n_splits
        self._predictors_scores = scores.sort_values(by = 'cv_score', 
                                                     ascending = False)
    
    def select_predictors(self, predictors_to_use = None):
        if isinstance(predictors_to_use, str):
            predictors_to_use = [predictors_to_use]
        if predictors_to_use is None:
            predictors_to_use = self._predictors_scores.index.tolist()
        try:
            predictors_with = self._predictors.index.tolist()
        except AttributeError:
            predictors_with = None
            self._predictors = pd.DataFrame(columns = ['with', 'without'])
        if predictors_with is None:
            predictors_with = []
        if len(predictors_with) == 0:
            initial_score = .5
        else:
            initial_score = np.max((self._predictors.iloc[-1, :]['with'], 
                                    self._predictors.iloc[-1, :]['without']))
        
        for n, p in tqdm_notebook(enumerate(predictors_to_use),
                                  total = len(predictors_to_use)):
            predictors_with.append(p)
            with_score = 0
            for split in range(self._n_splits):
                with_score += self._predict_for_split(split = split, 
                                                      predictors = predictors_with) / self._n_splits
            if n == 0:
                without_score = initial_score
            else:
                without_score = np.max((self._predictors.iloc[-1, :]['with'], 
                                        self._predictors.iloc[-1, :]['without']))
            self._predictors.loc[p, 'with'] = with_score
            self._predictors.loc[p, 'without'] = without_score
            if without_score > with_score:
                predictors_with.remove(p)

    @property
    def predictors(self):
        return self._predictors.copy()
    
    def validate(self, predictors_to_use = None, metric = roc_auc_score):
        if predictors_to_use is None:
            try:
                predictors = self._predictors
            except AttributeError:
                predictors = self._data.columns.tolist()
        else:
            predictors = predictors_to_use
        scores, si, gi = [], [], []
        for split in trange(self._n_splits):
            scores.append(self._predict_for_split(split = split, 
                                                  predictors = predictors))
            try:
                si.append(self._model.booster_.feature_importance(importance_type='split'))
                gi.append(self._model.booster_.feature_importance(importance_type='gain'))
            except:
                pass
        self._cv_score = scores  
        try:
            self.feat_imp = list(zip(si, gi))
        except:
            pass
    @property
    def cv_score(self):
        result = 'score = {:.6f} +- {:.3f}'.format(np.mean(self._cv_score),
                                                   np.std(self._cv_score,
                                                          ddof = 1))
        return (result, self._cv_score)
    
    def predict(self, predictors_to_use = None):
        if predictors_to_use is None:
            try:
                predictors = self._predictors
            except AttributeError:
                predictors = self._data.columns.tolist()
        else:
            predictors = predictors_to_use
        tr_preds = pd.DataFrame(index = self._train)
        va_preds = pd.DataFrame(index = self._train)
        te_preds = pd.DataFrame(index = pd.Index(data = self._test, 
                                                 name = 'SK_ID_CURR'))
        scores, si, gi = [], [], []
        for split in trange(self._n_splits):
            predicted = self._predict_for_split(split = split, 
                                                predictors = predictors,
                                                predict = True)
            scores.append(predicted[0])
            try:
                si.append(self.model.booster_.feature_importance(importance_type='split'))
                gi.append(self.model.booster_.feature_importance(importance_type='gain'))
            except:
                pass
            va_preds.loc[self._valid_cv[split], 'valid_' + str(split)] = predicted[1]
            te_preds.loc[:, 'test_' + str(split)] = predicted[2]           
        self._train_pred = tr_preds
        self._valid_pred = va_preds
        self._test_pred = te_preds
        self._submission = te_preds.mean(axis = 1).to_frame('TARGET')
        self._cv_score = scores  
        self.feat_imp = list(zip(si, gi))
        
    @property
    def valid_predictions(self):
        return self._valid_pred.copy()
    @property
    def test_predictions(self):
        return self._test_pred.copy()
    @property
    def submission(self):
        return self._submission.copy()
    
    def pieplots(self,column,
                 for_test = True,
                 for_target = True,
                 nan_as_cat = True):
        n_subplots = 2
        subtitles = [column + '_train']
        data = [self._data.loc[self._train, column].value_counts(normalize = True, dropna = nan_as_cat).values]
        labels = [self._data.loc[self._train, column].value_counts(normalize = True, dropna = nan_as_cat).index]
        if for_test:
            n_subplots += 1
            subtitles.append(column + '_test')
            data.append(self._data.loc[self._test, column].value_counts(normalize = True, dropna = nan_as_cat).values)
            labels.append(self._data.loc[self._test, column].value_counts(normalize = True, dropna = nan_as_cat).index)   
        if for_target:
            for t in [0, 1]:
                n_subplots += 1
                subtitles.append(column + ' target=' + str(t))
                idx = self._target[self._target.eq(t)].index
                data.append(self._data.loc[idx, column].value_counts(normalize = True, dropna = nan_as_cat).values)
                labels.append(self._data.loc[idx, column].value_counts(normalize = True, dropna = nan_as_cat).index)    
        f, ax = plt.subplots(1, n_subplots, 
                             figsize = (n_subplots * 4 + sum([for_test, for_target, for_target]), 4))
        for i in range(n_subplots - 1):
            ax[i].pie(x = data[i], 
                      labels = labels[i],
                      explode = [0.1] * len(labels[i]),
                      autopct = '%1.2f%%')
            ax[i].set(title = subtitles[i])
        plt.delaxes(ax[-1])
        plt.show()
        
    def distplots(self, column,
                  for_test = True,
                  for_target = True):
        if for_target:
            n_subplots = 3
        else:
            n_subplots = 2
        f, ax = plt.subplots(1, n_subplots, figsize = (n_subplots * 6, 4), 
                             sharex = True, sharey = True)
        d1 = sb.distplot(self._data.loc[self._train, column],
                         hist=False, label='train', ax = ax[0])
        if for_test:
            d2 = sb.distplot(self._data.loc[self._test, column], 
                             hist=False, label='test', ax = ax[0])
        plt.legend()
        if for_target:
            d3 = sb.distplot(self._data.loc[self._target[self._target.eq(0)].index, column], 
                             hist=False, label='target=0', ax = ax[1])
            d4 = sb.distplot(self._data.loc[self._target[self._target.eq(1)].index, column], 
                             hist=False, label='target=1', ax = ax[1])
            plt.legend()
        plt.delaxes(ax[-1])
        plt.show()
        
    def boxplots(self, column,
                 for_test = True,
                 for_target = True):
        n_subplots = 2
        if for_test:
            n_subplots += 1
        if for_target:
            n_subplots += 1
        f, ax = plt.subplots(1, n_subplots, figsize = (4 * (n_subplots + 1), 4), sharey = True)
        train = self._data.loc[self._train, [column]]
        b1 = sb.boxplot(data = train, y = column, orient = 'v', ax = ax[0])
        ax[0].set(title = 'train')
        if for_test:
            test = self._data.loc[self._test, [column]]
            b2 = sb.boxplot(data = test, y = column, orient = 'v', ax = ax[1])
            ax[1].set(title = 'test')
        if for_target:
            if for_test:
                i = 2
            else:
                i = 1
            train = self._data.loc[self._train, [column]].join(self._target.to_frame('TARGET'))
            b3 = sb.boxplot(data = train, y = column, x = 'TARGET', orient = 'v', ax = ax[i])
            ax[i].set(title = 'train')
        plt.delaxes(ax[-1])
        plt.show()
        
    def barplots(self, column,
                 for_test = True,
                 for_target = True,
                 nan_as_cat = True):
        if for_test:
            n_subplots = 3
        else:
            n_subplots = 2
        f, ax = plt.subplots(1, n_subplots, figsize = (n_subplots * 6, 4))        
        data = self._data.loc[self._train, column].value_counts(normalize = True, dropna = nan_as_cat).sort_index()
        pos = np.arange(data.shape[0])            
        if for_target:
            dt = []
            for t in [0, 1]:          
                idx = self._target[self._target.eq(t)].index
                sh = self._data.loc[self._train, column].shape[0]
                temp = self._data.loc[idx, column].value_counts(dropna = nan_as_cat) / sh
                dt.append(pd.merge(data.to_frame('total'),
                                   temp.to_frame(str(t)),
                                   left_index = True,
                                   right_index = True,
                                   how = 'left')[str(t)].fillna(0))
            
            b1 = ax[0].bar(pos, dt[0].values , alpha = 0.5)
            b2 = ax[0].bar(pos, dt[1].values, bottom = dt[0].values, alpha = 0.5)
            ax[0].legend([b1, b2], ['target=0', 'target=1'])
        else:
            b1 = ax[0].bar(pos, data.values , alpha = 0.5)     
        ax[0].set(title = column + ' train')
        ax[0].set_xticks(pos)
        ax[0].set_xticklabels(list(data.index), rotation = 90)
        if for_test:  
            data = pd.merge(data.to_frame('train'),
                            self._data.loc[self._test, column].value_counts(normalize = True, 
                                                                            dropna = nan_as_cat).to_frame('test'),
                            left_index = True,
                            right_index = True,
                            how = 'outer').fillna(0).sort_index(axis = 0)
            pos = np.arange(data.shape[0])
            b1 = ax[1].bar(pos, data.train.values, alpha = 0.5)
            b2 = ax[1].bar(pos, data.test.values, alpha = 0.5)
            ax[1].set(title = column + ' test')
            ax[1].set_xticks(pos)
            ax[1].set_xticklabels(list(data.index), rotation = 90)
            ax[1].legend([b1, b2], ['train', 'test'])
        plt.delaxes(ax[-1])    
        plt.show()
        
    def draw_average_default_level(self, column):
        data = pd.DataFrame()
        data[column] = self.data[column][self._train]
        data['TARGET'] = self._target
        stats = data.groupby(column).agg({'TARGET': ['mean', 'sum']})
        stats.columns = pd.Index(['mean', 'sum'])
        labels = list(stats.index)
        values = stats['mean'].values
        sizes = stats['sum'].values
        pos = np.arange(stats.shape[0])
        plt.figure(figsize = (np.max((len(labels), 6)), 4))
        plt.title(column + ' default level')
        plt.scatter(pos, values, s=sizes, c = range(len(labels)), alpha = 0.3)
        plt.scatter(pos, values, marker = '.', c = range(len(labels)))
        plt.xticks(pos, labels, rotation = 90 * (len(labels) > 2))
        plt.yticks([data.TARGET.mean()])
        plt.plot(pos, [data.TARGET.mean()] * len(labels), linestyle='dashed', c = 'r')
        plt.show()
        
        
def columns_classification(df, columns):
    fc, noc, nuc = [], [], []
    for column in columns:
        if df[column].nunique(dropna = False) < 3:
            fc.append(column)
        elif df[column].dtype == object:
            noc.append(column)
        else:
            nuc.append(column)           
    return fc, noc, nuc

def arr_to_txt(filename, arr):
    with open(filename + '.txt', 'w') as f:
        f.write(', '.join(map(str, arr)))

def arr_from_txt(filename):
    with open(filename + '.txt', 'r') as f:
        return f.read().split(', ')

def iv_verdict(iv):
    if iv <= 0.02:
        verdict = 'useless'
    elif iv <= 0.1:
        verdict = 'weak'
    elif iv <= 0.3:
        verdict = 'medium'
    elif iv <= 0.5:
        verdict = 'strong'
    else:
        verdict = 'suspicious'
    return verdict
