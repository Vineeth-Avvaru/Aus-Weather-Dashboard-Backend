import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats

def process_data(file_name):
    df = pd.read_csv(file_name)
    df[["year", "month", "day"]] = df["Date"].str.split("-", expand=True)
    df.drop(['Date','RainToday','RainTomorrow'],axis=1,inplace=True)
    locs = df['Location'].unique()
    for i in locs:
        if df[df['Location']==i]['WindGustDir'].shape[0]==df[df['Location']==i]['WindGustDir'].isnull().sum():
            continue
        df.loc[df['Location']==i,'WindGustDir']=df[df['Location']==i]['WindGustDir'].fillna(df[df['Location']==i]['WindGustDir'].mode()[0])
        df.loc[df['Location']==i,'WindDir9am']=df[df['Location']==i]['WindDir9am'].fillna(df[df['Location']==i]['WindDir9am'].mode()[0])
        df.loc[df['Location']==i,'WindDir3pm']=df[df['Location']==i]['WindDir3pm'].fillna(df[df['Location']==i]['WindDir3pm'].mode()[0])
    df['WindGustDir'] = df['WindGustDir'].fillna(df['WindGustDir'].mode()[0])
    df['WindDir9am'] = df['WindDir9am'].fillna(df['WindDir9am'].mode()[0])
    df['WindDir3pm'] = df['WindDir3pm'].fillna(df['WindDir3pm'].mode()[0])
    numerical = df.select_dtypes(exclude=['object']).columns
    categorical = df.select_dtypes(include=['object']).columns
    # Multiple Imputation by Chained Equations
    from sklearn.experimental import enable_iterative_imputer
    from sklearn.impute import IterativeImputer
    MiceImputed = df[numerical].copy(deep=True)
    print()
    mice_imputer = IterativeImputer()
    MiceImputed.iloc[:, :] = mice_imputer.fit_transform(MiceImputed)
    df[numerical] = MiceImputed
    df[['day','month','year']] = df[['day','month','year']].astype(np.int)
    df['WindSpeed'] = (df['WindSpeed9am'] + df['WindSpeed3pm'])/2
    df['Humidity'] =  (df['Humidity9am'] + df['Humidity3pm'])/2
    df['Pressure'] =  (df['Pressure9am'] + df['Pressure3pm'])/2
    df['Cloud'] =  (df['Cloud9am'] + df['Cloud3pm'])/2
    df['Temperature'] =  (df['Temp9am'] + df['Temp3pm'])/2
    df.drop(['WindSpeed9am','WindSpeed3pm','Humidity9am','Humidity3pm','Pressure9am','Pressure3pm','Cloud9am','Cloud3pm','Temp9am','Temp3pm'],axis=1,inplace=True)
    df_num = pd.concat([df[df.select_dtypes(exclude=['object']).columns],df['Location']],axis=1)
    df_cat = pd.concat([df[df.select_dtypes(include=['object']).columns], df_num[['year','month']]],axis=1 )
    df_num = df_num.groupby(['Location','year','month']).mean().reset_index()
    df_cat = df_cat.groupby(['Location','year','month']).agg(lambda x: stats.mode(x)[0]).reset_index()
    df_new = pd.concat([df_num,df_cat],axis=1)
    df_new = df_new.loc[:,~df_new.columns.duplicated()]
    df_new.drop(['day'],axis=1,inplace=True)
    data = df_new[df_new.select_dtypes(exclude=['object']).columns]
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    standardScaler = StandardScaler()
    data_norm=standardScaler.fit_transform(data.values)
    pca = PCA()
    pca_val = pca.fit_transform(data_norm)
    #loadings = pca.loadings
    eigen_vectors=pca.components_
    eigen_values = pca.singular_values_
    variance_explained = pca.explained_variance_ratio_
    cum_variance_explained=np.cumsum(variance_explained)
    # from sklearn.cluster import KMeans
    # error = []
    # for i in range(2, 15):
    #     kmeans = KMeans(n_clusters=i)
    #     kmeans.fit(data)
    #     error.append(kmeans.inertia_)

    # plt.plot(range(2, 15), error, color='b')
    # plt.grid(True)
    # plt.xlabel('No of clusters')
    # plt.ylabel('Errors')
    # plt.title('Elbow plot for K-Means clustering')
    # plt.show()
    km = KMeans(n_clusters=6)
    km.fit(data)
    data_cluster = km.labels_
    cluster_sizes = np.bincount(km.labels_)
    df_new['cluster'] = km.labels_
    df_new.insert(loc=0, column='index', value=df_new.index.values)
    return df_new.columns,df_new.values