import sklearn
from sklearn.metrics import classification_report, f1_score, accuracy_score, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold ,cross_val_score, train_test_split, cross_val_predict
from sklearn.tree import DecisionTreeClassifier 
from sklearn.model_selection import learning_curve
#from pandas_ml import ConfusionMatrix
import re
import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
import matplotlib.style as style

import pandas.api.types as pdtypes

from plotnine import *
from plydata import *

np.random.seed(1234)
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split

def read_process_data(data_path):
    print('reading datafile')
    data = pd.read_csv(data_path+'data.csv')
    data['CowID'] = data['MT_Results Individual_CowID']
    ### fill missing values and correct some values 
    print ('fill missing values and correct some values')
    data['HerdSurvey_Q6_How_often_cull_dairy_cows'].fillna('0', inplace = True)
    data['HerdSurvey_Q12_Herds_condemnedPercent'].fillna('0', inplace = True)
    data['HerdSurvey_Q13_injections_within2-3 weeks_Percent'].fillna('0', inplace = True)
    data['HerdSurvey_Q18A_Antibiotic_one'].fillna('None_reported', inplace = True)
    data['HerdSurvey_Q18B_Antibiotic_two'].fillna('None_reported', inplace = True)
    data['HerdSurvey_Q18C_Antibiotic_three'].fillna('None_reported', inplace = True)
    data['HerdSurvey_Q22_How_frequentlyMonthly'].fillna('0', inplace = True)
    data.replace('Variable (10/20x/m)', 'Variable', inplace = True)

    data['Ecoli resistance_Number of drugs Resistant'].fillna(0, inplace = True)
    data['Ecoli resistance_Resistant  to at least 1 drug (1=yes; 0=no)'].fillna(0, inplace = True)
    data['Ecoli resistance_Resistant  to at least 1 drug (1=yes; 0=no)'].fillna(0, inplace = True)
    data['Ecoli resistance_Multidrug Resistant (1=yes; 0=no)'].fillna(0,inplace = True)

    data['MT_Results Individual F_AMR_Number of drugs Resistant'].fillna(0,inplace = True)
    data['Entero resistance_Number of drugs Resistant'].fillna(0,inplace = True)
    data.rename(columns={'EcoliResist_AUG2*':'EcoliResist_AUG2'}, inplace=True)
    
    ### changing the column names 
    print ('changing the column names')
    conversion_dictionary = {
        'HerdSurvey_Q1_avg_num_milk_cows': 'HerdSize',
        'HerdSurvey_Q2_ Herd_rolling_milk_production_lbs': 'RollingHerdAvg',
        'HerdSurvey_Q3A_ Holstein':'Holstein',
        'HerdSurvey_Q3B_Jersey':'Jersey',
        'HerdSurvey_Q5_DairyHerdCullingPercentPerMonth':'CullPctMonth',
        'HerdSurvey_Q6_How_often_cull_dairy_cows':'CullTimesMonth',
        'HerdSurvey_Q7B_Disease':'MainCullReason',
        'HerdSurvey_Q8_Percent_Culled_dairy_sold_beef':'PctCullBeef',
        'HerdSurvey_Q12_Herds_condemnedPercent':'PctCullCondemned',
        'HerdSurvey_Q13_injections_within2-3 weeks_Percent':'PctInject',
        'HerdSurvey_Q14A_Veterinary':'VetTreats',
        'HerdSurvey_Q14B_Dairy Manager':'ManagerTreats',
        'HerdSurvey_Q14C_ Staff':'StaffTreats',
        'HerdSurvey_Q15A_Avoid_Drug':'ResiduePrevent',
        'HerdSurvey_Q16B_Chalk_Marks':'Chalk4Withdrawal',
        'HerdSurvey_Q17_record_of_drugs':'Inventory',
        'HerdSurvey_Q18A_Antibiotic_one':'Antibiotic1',
        'HerdSurvey_Q18B_Antibiotic_two':'Antibiotic2',
        'HerdSurvey_Q18C_Antibiotic_three':'Antibiotic3',
        'HerdSurvey_Q19A_Separately':'SeparateUse',
        'HerdSurvey_Q19B_Combine':'CombinationUse',
        'HerdSurvey_Q20B_Dose':'TrackAntibioticDose',
        'HerdSurvey_Q20C_Route':'TrackAntibioticRoute',
        'HerdSurvey_Q21_extra_label_drug_use':'FamiliarELDU',
        'HerdSurvey_Q22_How_frequentlyMonthly':'FreqELDU',
        'HerdSurvey_Q22A_Do Not Use':'NoELDU',
        'HerdSurvey_Q26_How_many_culled_':'NumberCulled',
        'HerdSurvey_Q27_SRP':'SalmonellaVaccine',
        'CowSurvey_Q1A_Low_Milk':'LowMilkCull',
        'CowSurvey_Q1B_Poor_Reproduction':'ReproCull',
        'CowSurvey_Q1C_Lameness':'LameCull',
        'CowSurvey_Q1E_Mastitis':'MastitisCull',
        'CowSurvey_Q1G_Other':'OtherCull',
        'CowSurvey_Q2_Antibiotics':'AMD',
        'CowSurvey_Q4A_Anti-Inflam':'Ani-Inf',
        'CowSurvey_Q4C_No Treat':'No-Treatment',
        'CowSurvey_Q4D_Other':'Other',
    }

    data = data.rename(columns = conversion_dictionary)
    
    data['Date'] = pd.to_datetime(data['Date'])
    mapping = {1: 'Winter', 2: 'Winter', 3: 'Spring', 4: 'Spring', 5: 'Spring', 6: 'Summer', 7: 'Summer', 8: 'Summer', 
               9: 'Autumn', 10: 'Autumn', 11: 'Autumn', 12: 'Winter'}
    data['Season'] = data['Date'].dt.month.map(mapping) 
    
    ### generating output variables 
    print ('generating output variables')
    Sal_drugs = ['SR_FOX','SR_AZI','SR_CHL','SR_TET','SR_AXO',
    'SR_AUG2','SR_CIP','SR_GEN','SR_Nal','SR_XNL',
    'SR_FIS','SR_SXT','SR_AMP','SR_STR']

    def Sal_R(c):
        drugs_R =[]
        for d in Sal_drugs:
            if c[d] == 'R' or c[d] == 'I':
                drug = d.replace('SR_', '')
                drugs_R.append(drug)
        return drugs_R

    data['Sal_DrugsR'] = data.apply(Sal_R, axis=1)

    Ecoli_drugs = ['EcoliResist_FOX','EcoliResist_AZI','EcoliResist_CHL','EcoliResist_TET','EcoliResist_AXO',
                 'EcoliResist_AUG2','EcoliResist_CIP','EcoliResist_GEN','EcoliResist_Nal','EcoliResist_XNL',
                 'EcoliResist_FIS','EcoliResist_SXT','EcoliResist_AMP','EcoliResist_STR','EcoliResist_KAN']

    def Ecoli_R(c):
        drugs_R =[]
        for d in Ecoli_drugs:
            if c[d] == 'R' or c[d] == 'I':
                drug = d.replace('EcoliResist_', '')
                drugs_R.append(drug)
        return drugs_R

    data['Ecoli_DrugsR'] = data.apply(Ecoli_R, axis=1)


    Enero_drugs = ['EnteroResist_TGC','EnteroResist_TET','EnteroResist_CHL','EnteroResist_DAP',
                 'EnteroResist_STR','EnteroResist_TYLT','EnteroResist_SYN','EnteroResist_LZD',
                 'EnteroResist_NIT', 'EnteroResist_PEN','EnteroResist_KAN','EnteroResist_ERY',
                 'EnteroResist_CIP','EnteroResist_VAN','EnteroResist_LIN','EnteroResist_GEN']

    def Entero_R(c):
        drugs_R =[]
        for d in Enero_drugs:
            if c[d] == 'R' or c[d] == 'I':
                drug = d.replace('EnteroResist_', '')
                drugs_R.append(drug)
        return drugs_R

    data['Entero_DrugsR'] = data.apply(Entero_R, axis=1)
    
    
    
    entero_drugs = list(set([a for b in data.Entero_DrugsR.tolist() for a in b]))
    ecoli_drugs = list(set([a for b in data.Ecoli_DrugsR.tolist() for a in b]))
    sal_drugs = list(set([a for b in data.Sal_DrugsR.tolist() for a in b]))
    ab_ab = sorted(list(set(sal_drugs+ ecoli_drugs+entero_drugs)))

    names = [
        'Ampicillin', 'Augmentin', 'Ceftriaxone', 'Chloramphenicol',
        'Ciprofloxacin', 'Erythromycin', 'Cefoxitin', 'Gentamicin', 'Kanamycin',
        'Lincomycin', 'Linezolid', 'Nitrofurantoin','Nalidixic acid', 'Streptomycin',
        'Trimethoprim-sulphamethoxazole', 'Synercid', 'Tetracycline',
        'Tylosin tartrate', 'Vancomycin', 'Ceftiofur'
    ]
    drug_class = [
        'Penicillins', 'Penicillins',  'Cephalosporins', 'Amphenicols',
        'Fluoroquinolones', 'Macrolides', 'Cephalosporins', 'Aminoglycosides', 'Aminoglycosides',
        'Lincosamides', 'Oxazolidinones', 'Nitrofuran antibacterial', 'Quinolones', 'Aminoglycosides', 
        'Folate pathway antagonist', 'Streptogramin', 'Tetracyclines', 
        'Macrolides', 'Glycopeptides', 'Cephalosporins'
    ]

    d = pd.DataFrame({'Abb': ab_ab, 'Antibiotic': names, 'drug_class':drug_class })
    inherent_ecoli = ['Lincosamides', 'Oxazolidinones', 'Penicillins', 'Streptogramins','Glycopeptides']
    inherent_entero = ['Cephalosporins', 'Lincosamides', 'Fluoroquinolones', 'Aminoglycosides', 'Aminocyclitols', 
                      'Sulfonamides', 'Folate pathway antagonist']
    inherent_sal = ['Cephalosporins', 'Aminoglycosides', 'Lincosamides', 'Oxazolidinones', 'Aminoglycosides', 'Glycopeptides']
    def entero_natural (c):
        if c['drug_class'] in inherent_entero:
            return 1
        else:
            return 0
    d['entero_natural'] = d.apply(entero_natural, axis = 1)

    def ecoli_natural (c):
        if c['drug_class'] in inherent_ecoli:
            return 1
        else:
            return 0
    d['ecoli_natural'] = d.apply(ecoli_natural, axis = 1)
    
    def sal_natural (c):
        if c['drug_class'] in inherent_sal:
            return 1
        else:
            return 0
    d['sal_natural'] = d.apply(sal_natural, axis = 1)
    
    
    inherent_ecoli_abb = d[d.ecoli_natural == 1].Abb.unique()
    inherent_entero_abb = d[d.entero_natural == 1].Abb.unique()
    inherent_sal_abb = d[d.sal_natural == 1].Abb.unique()

    name_dict = dict(d[['Abb', 'Antibiotic']].values)
    class_dict = dict(d[['Antibiotic', 'drug_class']].values)

    data['Entero_DrugsR_t'] = data.Entero_DrugsR.apply(lambda x: [i for i in x if i not in inherent_entero_abb])
    data['Ecoli_DrugsR_t'] = data.Ecoli_DrugsR.apply(lambda x: [i for i in x if i not in inherent_ecoli_abb])
    data['Sal_DrugsR_t'] = data.Sal_DrugsR.apply(lambda x: [i for i in x if i not in inherent_sal_abb])

    def get_ab_names_ecoli(c):
        return list(set([name_dict.get(k) for k in c.Ecoli_DrugsR_t]))
    data['Ecoli_AbR'] = data.apply(get_ab_names_ecoli, axis = 1)

    def get_ab_names_entero(c):
        return list(set([name_dict.get(k) for k in c.Entero_DrugsR_t]))
    data['Entero_AbR'] = data.apply(get_ab_names_entero, axis = 1)

    def get_ab_names_sal(c):
        return list(set([name_dict.get(k) for k in c.Sal_DrugsR_t]))
    data['Sal_AbR'] = data.apply(get_ab_names_sal, axis = 1)

    def get_ab_class_ecoli(c):
        return list(set([class_dict.get(k) for k in c.Ecoli_AbR]))
    data['Ecoli_AbR'] = data.apply(get_ab_class_ecoli, axis = 1)

    def get_ab_class_entero(c):
        return list(set([class_dict.get(k) for k in c.Entero_AbR]))
    data['Entero_AbR'] = data.apply(get_ab_class_entero, axis = 1)

    def get_ab_class_sal(c):
        return list(set([class_dict.get(k) for k in c.Sal_AbR]))
    data['Sal_AbR'] = data.apply(get_ab_class_sal, axis = 1)

    def compile_all(c):
        return list(set(c['Sal_AbR']+c['Entero_AbR']+c['Ecoli_AbR']))

    data['R'] = data.apply(compile_all, axis=1)
    data['R_n'] = data['R'].str.len()


    def compile_all_commensal(c):
        return list(set(c['Entero_AbR']+c['Ecoli_AbR']))

    data['R_commensal'] = data.apply(compile_all_commensal, axis=1)
    data['R_commensal_n'] = data['R_commensal'].str.len()
    data['Sal_AbR_n'] = data['Sal_AbR'].str.len()
    data['Entero_AbR_n'] = data['Entero_AbR'].str.len()
    data['Ecoli_AbR_n'] = data['Ecoli_AbR'].str.len()

    #### final R_factor output
    print('final R_factor output')
    conditions  = [ 
        data['Sal_AbR_n'] + data['Entero_AbR_n'] +data['Ecoli_AbR_n'] ==0,
        data['Sal_AbR_n'] + data['Entero_AbR_n'] +data['Ecoli_AbR_n'] <=2,
        (data['Sal_AbR_n']>2) | (data['Entero_AbR_n']>2) | (data['Ecoli_AbR_n']>2), 
        (data['Sal_AbR_n'] == 2) | (data['Entero_AbR_n'] == 2) | (data['Ecoli_AbR_n'] == 2), 
        (data['Sal_AbR_n'] == 1) | (data['Entero_AbR_n'] == 1) | (data['Ecoli_AbR_n'] == 1)
    ]

    choices = [0, 1, 2, 1, 1]
    data['R_factor'] = np.select(conditions, choices, default= 0)
    data['R_factor'] = data['R_factor'].astype('category')
    
    conditions = [
        (data['Sal_AbR_n'] ==0),
        (data['Sal_AbR_n'] <=2),
        (data['Sal_AbR_n'] >2)]
    choices = [0, 1, 2]
    data['Sal_R'] = np.select(conditions, choices, default= 0)
    data['Sal_R'] = data['Sal_R'].astype('category')
    
    conditions = [
        (data['Ecoli_AbR_n'] == 0),
        (data['Ecoli_AbR_n'] <=2),
        (data['Ecoli_AbR_n'] >2)]
    choices = [0, 1, 2]
    data['Ecoli_R'] = np.select(conditions, choices, default= 0)
    data['Ecoli_R'] = data['Ecoli_R'].astype('category')
    
    conditions = [
        (data['Entero_AbR_n'] == 0),
        (data['Entero_AbR_n'] <=2),
        (data['Entero_AbR_n'] >2)]
    choices = [0, 1, 2]
    data['Entero_R'] = np.select(conditions, choices, default= 0)
    data['Entero_R'] = data['Entero_R'].astype('category')
    
    conditions = [
        (data['R_commensal_n'] ==0),
        (data['R_commensal_n'] <=2),
        (data['R_commensal_n'] >2)]
    choices = [0, 1, 2]
    data['Commensal_R'] = np.select(conditions, choices, default= 0)
    data['Commensal_R'] = data['Commensal_R'].astype('category')
    data.replace('Excenel ', 'Excenel', inplace = True)
    data.replace('Spectramst', 'Spectramast', inplace = True)
    data.replace('Ceftioflex', 'Ceftiofur', inplace = True)
    data.replace('Naxcel', 'Ceftiofur', inplace = True)
    data.replace('Spectramast', 'Ceftiofur', inplace = True)
    data.replace('Excenel', 'Ceftiofur', inplace = True)
    data.replace('Polyflex', 'Penicillin', inplace = True)
    antibiotics = list(set(data.Antibiotic1.unique().tolist()+data.Antibiotic2.unique().tolist()+data.Antibiotic3.unique().tolist()))
    antibiotics.remove('None_reported')
    data['Antibiotics'] =data['Antibiotic1']+', ' + data['Antibiotic2']+', '+data['Antibiotic3'] 
    def create_antibiotic_column(c, a):
        if a in c['Antibiotics']:
            return 1
        else:
            return 0

    for a in antibiotics:
        data[a] = data.apply(create_antibiotic_column, a = a, axis=1)
        print(data[a].value_counts())
    
    return data, d