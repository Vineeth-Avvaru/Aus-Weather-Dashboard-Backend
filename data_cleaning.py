{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "import sys\n",
    "import os\n",
    "import cv2\n",
    "import numpy as np\n",
    "import matplotlib.pyplot as plt\n",
    "from scipy import ndimage, misc\n",
    "from IPython.display import display, Image\n",
    "from mpl_toolkits.mplot3d import Axes3D\n",
    "from matplotlib import cm\n",
    "import pandas as pd\n",
    "from scipy import stats"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "df = pd.read_csv('WeatherAUS.csv')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "df[[\"year\", \"month\", \"day\"]] = df[\"Date\"].str.split(\"-\", expand=True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "df.drop(['Date','RainToday','RainTomorrow'],axis=1,inplace=True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "locs = df['Location'].unique()\n",
    "for i in locs:\n",
    "    if df[df['Location']==i]['WindGustDir'].shape[0]==df[df['Location']==i]['WindGustDir'].isnull().sum():\n",
    "        continue\n",
    "    df.loc[df['Location']==i,'WindGustDir']=df[df['Location']==i]['WindGustDir'].fillna(df[df['Location']==i]['WindGustDir'].mode()[0])\n",
    "    df.loc[df['Location']==i,'WindDir9am']=df[df['Location']==i]['WindDir9am'].fillna(df[df['Location']==i]['WindDir9am'].mode()[0])\n",
    "    df.loc[df['Location']==i,'WindDir3pm']=df[df['Location']==i]['WindDir3pm'].fillna(df[df['Location']==i]['WindDir3pm'].mode()[0])\n",
    "df['WindGustDir'] = df['WindGustDir'].fillna(df['WindGustDir'].mode()[0])\n",
    "df['WindDir9am'] = df['WindDir9am'].fillna(df['WindDir9am'].mode()[0])\n",
    "df['WindDir3pm'] = df['WindDir3pm'].fillna(df['WindDir3pm'].mode()[0])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "numerical = df.select_dtypes(exclude=['object']).columns\n",
    "categorical = df.select_dtypes(include=['object']).columns"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Multiple Imputation by Chained Equations\n",
    "from sklearn.experimental import enable_iterative_imputer\n",
    "from sklearn.impute import IterativeImputer\n",
    "MiceImputed = df[numerical].copy(deep=True)\n",
    "print()\n",
    "mice_imputer = IterativeImputer()\n",
    "MiceImputed.iloc[:, :] = mice_imputer.fit_transform(MiceImputed)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "df[numerical] = MiceImputed\n",
    "df[['day','month','year']] = df[['day','month','year']].astype(np.int)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "df['WindSpeed'] = (df['WindSpeed9am'] + df['WindSpeed3pm'])/2\n",
    "df['Humidity'] =  (df['Humidity9am'] + df['Humidity3pm'])/2\n",
    "df['Pressure'] =  (df['Pressure9am'] + df['Pressure3pm'])/2\n",
    "df['Cloud'] =  (df['Cloud9am'] + df['Cloud3pm'])/2\n",
    "df['Temperature'] =  (df['Temp9am'] + df['Temp3pm'])/2\n",
    "df.drop(['WindSpeed9am','WindSpeed3pm','Humidity9am','Humidity3pm','Pressure9am','Pressure3pm','Cloud9am','Cloud3pm','Temp9am','Temp3pm'],axis=1,inplace=True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "df_num = pd.concat([df[df.select_dtypes(exclude=['object']).columns],df['Location']],axis=1)\n",
    "df_cat = pd.concat([df[df.select_dtypes(include=['object']).columns], df_num[['year','month']]],axis=1 )"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "df_num = df_num.groupby(['Location','year','month']).mean().reset_index()\n",
    "df_cat = df_cat.groupby(['Location','year','month']).agg(lambda x: stats.mode(x)[0]).reset_index()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "df_new = pd.concat([df_num,df_cat],axis=1)\n",
    "df_new = df_new.loc[:,~df_new.columns.duplicated()]\n",
    "df_new.drop(['day'],axis=1,inplace=True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "df_new"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "data = df_new[df_new.select_dtypes(exclude=['object']).columns]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.decomposition import PCA\n",
    "from sklearn.preprocessing import StandardScaler\n",
    "\n",
    "standardScaler = StandardScaler()\n",
    "data_norm=standardScaler.fit_transform(data.values)\n",
    "pca = PCA()\n",
    "pca_val = pca.fit_transform(data_norm)\n",
    "#loadings = pca.loadings\n",
    "eigen_vectors=pca.components_\n",
    "eigen_values = pca.singular_values_\n",
    "variance_explained = pca.explained_variance_ratio_\n",
    "cum_variance_explained=np.cumsum(variance_explained)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# from sklearn.cluster import KMeans\n",
    "# error = []\n",
    "# for i in range(2, 15):\n",
    "#     kmeans = KMeans(n_clusters=i)\n",
    "#     kmeans.fit(data)\n",
    "#     error.append(kmeans.inertia_)\n",
    "\n",
    "# plt.plot(range(2, 15), error, color='b')\n",
    "# plt.grid(True)\n",
    "# plt.xlabel('No of clusters')\n",
    "# plt.ylabel('Errors')\n",
    "# plt.title('Elbow plot for K-Means clustering')\n",
    "# plt.show()\n",
    "km = KMeans(n_clusters=6)\n",
    "km.fit(data)\n",
    "data_cluster = km.labels_\n",
    "cluster_sizes = np.bincount(km.labels_)\n",
    "df_new['cluster'] = km.labels_"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "\n",
    "# sampling_results = pd.DataFrame(columns=df.columns)\n",
    "# for i in range(4):\n",
    "#     cluster_size = cluster_sizes[i]\n",
    "#     cluster_records = df[data_cluster == i]\n",
    "#     sample_size = int(cluster_size * 0.03)\n",
    "#     sampling_results = pd.concat([sampling_results, cluster_records.iloc[random.sample(range(cluster_size), sample_size)]]).reset_index(drop=True)\n",
    "# df=sampling_results\n",
    "# data=df.values[:,:-1]    "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# plt.scatter(y=df_new['Rainfall'].values, x=df_new['WindDir3pm'].values)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# import numpy as np\n",
    "# import matplotlib.pyplot as plt\n",
    "# import seaborn as sns\n",
    "# MiceImputed=df_new\n",
    "# corr = MiceImputed.corr()\n",
    "# mask = np.triu(np.ones_like(corr, dtype=np.bool))\n",
    "# f, ax = plt.subplots(figsize=(20, 20))\n",
    "# cmap = sns.diverging_palette(250, 25, as_cmap=True)\n",
    "# sns.heatmap(corr, mask=mask, cmap=cmap, vmax=None, center=0,square=True, annot=True, linewidths=.5, cbar_kws={\"shrink\": .9})"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.8.5"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
