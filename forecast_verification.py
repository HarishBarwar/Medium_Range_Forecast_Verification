# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import pandas as pd
from tabulate import tabulate
#import warnings


class Medium_Range_Forecast_Verification:
    def __init__(self, forecast, observed, dv_threshold=None, climatology=None, start_date='', end_date='', dv_col=None, usability_analysis=None):
        self.pred=forecast
        self.obs=observed
        self.thr=dv_threshold
        self.s_dt=datetime.strptime(start_date, '%d-%m-%Y') if start_date else None
        self.e_dt=datetime.strptime(end_date, '%d-%m-%Y') if end_date else None
        self.pred.columns = self.pred.columns.str.upper() 
        self.obs.columns = self.obs.columns.str.upper()
        self.cl = climatology
        self.dv_col = dv_col
        self.ua = usability_analysis
        date_warn = False
        
        for pred_column, obs_column in zip(self.pred.columns, self.obs.columns):
            if pd.api.types.is_datetime64_any_dtype(self.pred[pred_column]) and pd.api.types.is_datetime64_any_dtype(self.obs[obs_column]):
                self.pred = self.pred.rename(columns={pred_column: 'DATE'})
                self.obs = self.obs.rename(columns={obs_column: 'DATE'})

                self.obs['DATE'] = pd.to_datetime(self.obs['DATE'])
                self.pred['DATE'] = pd.to_datetime(self.pred['DATE'])
                dates_set_obs = set(self.obs['DATE'])
                dates_set_pred = set(self.pred['DATE'])
                common_dates = dates_set_obs.intersection(dates_set_pred)
                self.obs = self.obs[self.obs['DATE'].isin(common_dates)]
                self.pred = self.pred[self.pred['DATE'].isin(common_dates)]
                self.pred = self.pred.groupby('DATE').mean().reset_index()
                self.obs = self.obs.groupby('DATE').mean().reset_index()

                
                if self.s_dt is None:
                    self.s_dt = self.obs['DATE'].min()
                    self.obs = self.obs.reset_index(inplace=True)
                    self.pred = self.pred.reset_index(inplace=True)
                    
                if self.e_dt is None:
                    self.e_dt = self.obs['DATE'].max()
                    self.obs = self.obs.reset_index(inplace=True)
                    self.pred = self.pred.reset_index(inplace=True)
                    
                if self.s_dt or self.e_dt:
                    # print(type(self.s_dt),type(self.e_dt))
                    # print(self.obs.columns)
                    self.obs = self.obs[(self.obs['DATE'] >= self.s_dt) & (self.obs['DATE'] <= self.e_dt)]
                    self.pred = self.pred[(self.pred['DATE'] >= self.s_dt) & (self.pred['DATE'] <= self.e_dt)]
                    self.obs = self.obs.reset_index(drop=True)
                    self.pred = self.pred.reset_index(drop=True)
                    # print(self.pred)
                break
            else:
                date_warn = True

        if not all(self.pred.columns == self.obs.columns):
            raise ValueError("Column names mismatch between predicted and observed data.")
        
        if date_warn:
            #warnings.warn("Provide data with date for accurate verification.", category=UserWarning)
            raise ValueError("Data without dates can potentially lead to illogical or inaccurate verification...")
        
                
        if self.cl is not None:
            if type(self.cl) is pd.core.frame.DataFrame and (len(self.cl.columns) == len(self.obs.columns)):
                for column in self.cl.columns:
                    if pd.api.types.is_datetime64_any_dtype(self.cl[column]):
                       self.cl = self.cl.drop(columns=[column])
                    # columns = self.obs.columns.drop('DATE')
                    self.cl.columns = self.cl.columns.str.upper()
                    # self.cl = self.cl.mean()
                    # self.cl = pd.DataFrame(self.cl, columns=columns)
            elif type(self.cl) is dict and (len(self.cl) == len(self.pred.columns.drop("DATE"))):
                self.cl = self.cl.columns.str.upper()
                self.cl = pd.DataFrame(self.cl)
            elif type(self.cl) is list and (len(self.cl) == len(self.pred.columns.drop("DATE"))):
                columns = self.obs.columns.drop('DATE')
                self.cl = pd.DataFrame([self.cl], columns=columns)
            else:
                raise ValueError('Invalid climatology value...') 
        
        self.mean_error()
        self.mse()
        self.rmse()
        self.mae()
        self.multiplicative_bias()
        self.corr_coeff()
        if self.cl is not None:
            self.anomaly_correlation()
        # else:
        #     self.anomaly_corr_values=[]
            
        if (self.thr is not None) & (self.dv_col is not None):
            self.dichotomous_verification()
            
        if (self.ua is not None):
            self.usability_analysis()

        
        
    def eyeball_verification(self):     
        for column in self.pred.columns.drop('DATE'):
            plt.figure(figsize=(10, 6))
            if 'DATE' in self.pred.columns and 'DATE' in self.obs.columns: 
                plt.plot(self.pred['DATE'], self.pred[column], label=f'Forecast {column}', marker='o')
                plt.plot(self.obs['DATE'], self.obs[column], label=f'Observed {column}', marker='x')
                plt.gcf().autofmt_xdate()
                plt.xlabel('Date')
            else:
                num_data_points1 = len(self.pred)
                num_data_points2 = len(self.obs)
                plt.plot(range(num_data_points1), self.pred[column], label=f'Forecast {column}', marker='o')
                plt.plot(range(num_data_points2), self.obs[column], label=f'Observed {column}', marker='x')
                num_data_points=np.array([num_data_points1, num_data_points2]).max()
                plt.xlabel(f'Number of Data Points ({num_data_points} days)')
                plt.text(0.5, 0.5, 'Provide dataset with date for accurate result', horizontalalignment='center', 
                     verticalalignment='center', transform=plt.gca().transAxes, fontsize=12, color='red')
            
            plt.ylabel(column)
            plt.title(f'Forecast vs. Observed for {column}')
            plt.legend()
            plt.grid(True)
            plt.show()

    def mean_error(self):
        self.m_error_values=[]
        for column in self.pred.columns.drop("DATE"):
            mean_error = np.mean(self.pred[column] - self.obs[column])
            self.m_error_values.append([column, mean_error])
        self.mean_error_table = tabulate(self.m_error_values, headers=['Column', 'Mean Error'], tablefmt='grid')
        return self.mean_error_table
        
    def mse(self):
        self.mse_values=[]
        for column in self.pred.columns.drop("DATE"):
            mse=((self.pred[column] - self.obs[column]) ** 2).mean()
            self.mse_values.append([column, mse])
        self.mse_table = tabulate(self.mse_values, headers=['Column', 'MSE'], tablefmt='grid')
        return self.mse_table
        
        
    def rmse(self):
        self.rmse_values=[]
        for column in self.pred.columns.drop("DATE"):
            rmse=np.sqrt(((self.pred[column] - self.obs[column]) ** 2).mean())
            self.rmse_values.append([column, rmse])
        self.rmse_table = tabulate(self.rmse_values, headers=['Column', 'RMSE'], tablefmt='grid')
        return self.rmse_table

    
    def mae(self):
        self.mae_values=[]
        for column in self.pred.columns.drop("DATE"):
            mae = np.mean(np.abs(self.pred[column] - self.obs[column]))
            self.mae_values.append([column, mae])
        self.mae_table = tabulate(self.mae_values, headers=['Column', 'MAE'], tablefmt='grid')
        return self.mae_table
        
    
    def multiplicative_bias(self):
        self.m_bias_values=[]
        for column in self.pred.columns.drop("DATE"):
            m_bias = (np.mean(self.pred[column])) / (np.mean(self.obs[column]))
            #m_bias = ((1/len(self.pred[column])) * (np.sum(self.pred[column]))) / ((1/len(self.obs[column])) * (np.sum(self.obs[column])))
            self.m_bias_values.append([column, m_bias])
        self.m_bias_table = tabulate(self.m_bias_values, headers=['Column', 'MAE'], tablefmt='grid')
        return self.m_bias_table
    
    
    def corr_coeff(self):
        self.corr_coeff_values = []
        for column in self.pred.columns.drop('DATE'):
            pred_mean = np.mean(self.pred[column])
            obs_mean = np.mean(self.obs[column])
            numerator = np.sum((self.pred[column] - pred_mean) * (self.obs[column] - obs_mean))
            denominator = np.sqrt(np.sum((self.pred[column] - pred_mean)**2) * np.sum((self.obs[column] - obs_mean)**2))   
            corr_coeff = numerator / denominator
            self.corr_coeff_values.append([column, corr_coeff])
        self.corr_coeff_table = tabulate(self.corr_coeff_values, headers=['Column', 'Correlation Coefficient'], tablefmt='grid')
        return self.corr_coeff_table
        
    
    def anomaly_correlation(self):
        if self.cl is None:
            raise ValueError("Anomaly correlation requires climatology")
        self.anomaly_corr_values = []
        for column in self.pred.columns.drop("DATE"):
            obs_anomaly = self.obs[column] - self.cl[column].mean()
            forecast_anomaly = self.pred[column] - self.cl[column].mean()
            anomaly_corr = np.corrcoef(obs_anomaly, forecast_anomaly)[0, 1]
            self.anomaly_corr_values.append([column, anomaly_corr])
        self.anomaly_corr_table = tabulate(self.anomaly_corr_values, headers=['Column', 'Anomaly Correlation'], tablefmt='grid')
        return self.anomaly_corr_table



    def dichotomous_verification(self, column_name=None, threshold=None):       
        if column_name is not None:
            self.dv_col = column_name

        if threshold is not None:
            self.thr = threshold        
        
        if (self.thr is not None) and (self.dv_col is not None):
            pass
        else:
            raise ValueError("Threshold value or column name is missing.")
    
        # tp = tn = fp = fn = 0
        scores=[]
        contin=''
        self.accuracy = []
        self.sensitivity = []
        self.specificity = []
        self.positive_predictive_value = []
        self.negative_predictive_value = []
        self.fp_rate = []
        self.bias = []
        self.false_alarm_ratio = []
        self.threat_score = []
        self.equitable_threat_score = []
        self.hanssen_kuipers = []
        self.heidke_skill_score = []
        self.odds_ratio = []
        self.odds_ratio_skill_score = []
        
        score_names = ['Accuracy', 'Sensitivity', 'Specificity', 'Positive Predictive Value', 
                'Negative Predictive Value', 'False Positive Rate', 'Bias', 
                'False Alarm Ratio', 'Threat Score', 'Equitable Threat Score', 
                'Hanssen-Kuipers Score', 'Heidke Skill Score', 'Odds Ratio', 
                'Odds Ratio Skill Score']
        # difference = self.pred - self.obs
        if (self.dv_col.upper() in self.pred.columns) & (self.dv_col.upper() in self.obs.columns):
            tp = tn = fp = fn = None
            if self.thr is not None:
                correct_forecast = (self.pred[self.dv_col.upper()] >= self.thr).astype(int)
                correct_observed = (self.obs[self.dv_col.upper()] >= self.thr).astype(int)
            else:
                raise ValueError("Invalid threshold values.")

            tp = hits = ((correct_forecast == 1) & (correct_observed == 1)).sum()
            tn = correct_negatives = ((correct_forecast == 0) & (correct_observed == 0)).sum()
            fp = false_alarms = ((correct_forecast == 1) & (correct_observed == 0)).sum()
            fn = misses = ((correct_forecast == 0) & (correct_observed == 1)).sum()

            contingency_table = pd.DataFrame({'Forecast Yes': [tp, fp],
                                              'Forecast No': [fn, tn]}, 
                                              index=['Observed Yes', 'Observed No'])
            contingency_table['Total'] = contingency_table.sum(axis=1)

            contingency_table.loc['Total'] = contingency_table.sum()
            contin+=(f"\nContingency Table for {self.dv_col}: \n{contingency_table}\n")

 
            
            self.accuracy.append((tp + tn) / (tp + tn + fp + fn))
            self.sensitivity.append(tp / (tp + fn))
            self.specificity.append(tn / (tn + fp))
            self.positive_predictive_value.append(tp / (tp + fp))
            self.negative_predictive_value.append(tn / (tn + fn))
            self.fp_rate.append(fp / (fp + tn))
            self.bias.append((tp + fp) / (tp + fn))
            self.false_alarm_ratio.append(fp / (tp + fp))
            self.threat_score.append(tp / (tp + fn + fp))
            hits_random = ((hits + misses) * (hits + false_alarms)) / (tp + tn + fp + fn)
            self.equitable_threat_score.append((hits - hits_random) / (hits + misses + false_alarms - hits_random))
            self.hanssen_kuipers.append(hits / (hits + misses) - false_alarms / (false_alarms + correct_negatives))
            expected_correct_random = (1/(len(self.pred[self.dv_col.upper()]))) * ((hits + misses) * (hits + false_alarms) + (correct_negatives + misses) * (correct_negatives + false_alarms))
            self.heidke_skill_score.append(((hits + correct_negatives) - expected_correct_random) / (len(self.pred[self.dv_col.upper()]) - expected_correct_random))
            self.odds_ratio.append((hits * correct_negatives) / (misses * false_alarms))
            self.odds_ratio_skill_score.append((hits * correct_negatives - misses * false_alarms) / (hits * correct_negatives + misses * false_alarms))
            
        scores= [self.accuracy, self.sensitivity, self.specificity, self.positive_predictive_value, 
                       self.negative_predictive_value, self.fp_rate, self.bias, self.false_alarm_ratio, 
                       self.threat_score, self.equitable_threat_score, self.hanssen_kuipers, 
                       self.heidke_skill_score, self.odds_ratio, self.odds_ratio_skill_score]

        scores_df1 = pd.DataFrame(scores, columns=[self.dv_col], index=score_names)
        scores_df = scores_df1
        head=['Verifications',self.dv_col]
        scores_table = tabulate(scores_df, headers=head,tablefmt='grid')
        self.contingency_tables=contin
        return scores_table

    

    def usability_analysis(self):
        correct=[]
        useable=[]
        unusable=[]
        for col in self.ua:
            cor=((self.pred[col]>=0) & (self.pred[col]<=(self.obs[col]+self.ua[col][0]))).astype(int).sum()
            use=((self.pred[col]>=(self.obs[col]+self.ua[col][0])) & (self.pred[col]<=(self.obs[col]+self.ua[col][1]))).astype(int).sum()
            unuse=(self.pred[col]>=(self.obs[col]+self.ua[col][1])).astype(int).sum()
            correct.append(cor)
            useable.append(use)
            unusable.append(unuse)
        
        usability_dict = {'Correct':correct,'Useable':useable,'Unusable':unusable}
        index_name = list(self.ua.keys())
        usability = pd.DataFrame(usability_dict, index=index_name)
        usability1 = usability.copy()
        usability['Total'] = usability.sum(axis=1)
        usability_percentage = usability.div(usability1.sum(axis=1), axis=0) * 100
        # self.usability_percentage['Total'] = usability_percentage.sum(axis=1)
        # self.usability_combined = usability.astype(str) + ' (' + usability_percentage.round(2).astype(str) + '%)'
        usability_table = tabulate(usability, headers=['Parameters']+list(usability_dict.keys())+['Total'], tablefmt='grid')
        

        
        self.usability_percentage_table = tabulate(usability_percentage, headers=['Parameters']+list(usability_dict.keys())+['Total'], tablefmt='grid')
        # usability_percentage_table[((float(usability_percentage_table['Correct']) + float(usability_percentage_table['Usable']))) < 50,'accuracy'] = 'Poor'
        # usability_percentage_table.loc[((float(usability_percentage_table['Correct']) + float(usability_percentage_table['Usable']))) > 70, 'accuracy'] = 'Good'
        # usability_percentage_table.loc[((float(usability_percentage_table['Correct']) + float(usability_percentage_table['Usable']))) >= 50, 'accuracy'] = 'Moderate'
        # self.usability_percentage_table = pd.concat([usability_percentage_table, usability_percentage_table['accuracy']], axis=1)

        
        return usability_table
    
    
    
    def scores(self):
        headers = ['errors or coefficient'] + list(self.obs.columns.drop('DATE'))
        rows = []

        error_names = ['Mean Error', 'MSE', 'RMSE', 'MAE', 'Multiplicative Bias', 'Correlation Coefficient', 'Anomaly Correlation']
        error_values = [self.m_error_values, self.mse_values, self.rmse_values, self.mae_values, self.m_bias_values, self.corr_coeff_values, self.anomaly_corr_values]

        for error_name, error_value in zip(error_names, error_values):
            row = [error_name]

            for column, value in zip(self.pred.columns.drop("DATE"), error_value):
                row.append(value[1])

            rows.append(row)

        return (tabulate(rows, headers=headers, tablefmt='grid'))
        
    def all_scores(self):
        verification_result=self.dichotomous_verification()
        a=120*'-'
        b=120*'='
        # a=41*'=x='
        s=50*" "
        s1=45*" "
        s2=38*' '
        print(f" {b}\n\n{s}Errors and Correlation:\n\n{self.scores()}\n\n {a.center(50)}\n\n{s1}Dichotomous verification of {self.dv_col}:\n\n{self.contingency_tables}\n\nDichotomous Verification Results:\n{verification_result}\n\n {a.center(50)}\n\n{s2}Usability Analysis of Different Weather Parameters: \n\n{self.usability_analysis()}\n\nPercentage Usability:\n{self.usability_percentage_table}\n\n {b}")



    


    