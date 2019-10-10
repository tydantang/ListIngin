#!/usr/bin/env python
# coding: utf-8

# ## Libraries and Packages

# In[2]:


# Python 3 environment 
import os

# Machine Learning Classifiers
from lightgbm import LGBMRegressor, Booster, cv as lgbcv, Dataset as lgbDs
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsClassifier

# Unsupervised Clustering
from sklearn.cluster import KMeans, DBSCAN
from sklearn.mixture import GaussianMixture

# Evaluation Tools
from sklearn.feature_selection import RFECV
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import KFold, cross_val_score, cross_val_predict, StratifiedKFold, GridSearchCV
 # for K-fold cross validation
import time
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve

# Processing and Visualization Libraries
import numpy as np
import pandas as pd
from IPython.display import Image
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn import preprocessing
from sklearn import metrics
from matplotlib.colors import ListedColormap
import graphviz
import zipfile
import random

import warnings
warnings.filterwarnings('ignore')

# New tools
import gzip
import m2cgen as m2c
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from scipy.interpolate import make_interp_spline, BSpline


# In[3]:


# Allow Jupyter Notebook to display all output in the cell not just the last result
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"


# Max rows/columns to display

# In[4]:


pd.options.display.max_rows = 200
pd.set_option('display.max_colwidth', -1)
pd.set_option('display.max_columns', 999)


# In[5]:


# Unzip gdata
file1 = gzip.open('D:\Insight DS\Airbnb\April\listings.csv.gz', 'rb')
file2 = gzip.open('D:\Insight DS\Airbnb\May\listings.csv.gz', 'rb')
file3 = gzip.open('D:\Insight DS\Airbnb\June\listings.csv.gz', 'rb')
file4 = gzip.open('D:\Insight DS\Airbnb\July\listings.csv.gz', 'rb')

l_df1 =pd.read_csv(file1)
l_df2 =pd.read_csv(file2)
l_df3 =pd.read_csv(file3)
l_df4 =pd.read_csv(file4)


# In[6]:


l_df = pd.concat([l_df1, l_df2, l_df3, l_df4], ignore_index=True)

file1.close()
file2.close()
file3.close()
file4.close()


# In[7]:


# Be careful with the data type 
l_df['price'].dtype
# l_df['price'].max()
l_df['price'].replace( '[\$,)]','', regex=True ).replace( '[(]','-',   regex=True ).astype(float).max()


# In[8]:


# from pivottablejs import pivot_ui # a bit messy still
# pivot_ui(l_df)


# In[9]:


# sns.distplot(l_df['price'].replace( '[\$,)]','', regex=True )
#                .replace( '[(]','-',   regex=True ).astype(float), kde=False)


# In[10]:


# sns.distplot(l_df['review_scores_rating'].dropna())


# In[11]:


# sns.distplot(l_df['review_scores_cleanliness'].dropna())


# In[12]:


# fig, axs = plt.subplots(figsize=(6,4))
# axs.scatter(l_df['review_scores_rating'], l_df['price'].replace( '[\$,)]','', regex=True ).replace( '[(]','-',   regex=True ).astype(float))
# # axs.set_yticks(axs.get_yticks()[::10]) use this when tick labels overlap
# axs.set_ylim([0, 5000])
# plt.show()


# In[13]:


# # Select DataFrame Rows Based on multiple conditions on columns
# l_df1 = l_df[['review_scores_rating', 'review_scores_cleanliness', 'review_scores_', 'price'
#               ]][(l_df['review_scores_cleanliness'].notna())
#                  & (l_df['review_scores_rating'].notna()) &
#                  (l_df['price'].replace('[\$,)]', '', regex=True).replace(
#                      '[(]', '-', regex=True).astype(float) < 300)]
# l_df1['price'] = l_df['price'].replace('[\$,)]', '', regex=True).replace(
#     '[(]', '-', regex=True).astype(float)
# l_df1


# In[14]:


# F1 = 'bathrooms'
# F2 = 'bedrooms'
# F3 = 'beds'
# F4 = 'minimum_nights'
# F5 = 'maximum_nights'
# F6 = 'availability_30'
# F7 = 'availability_60'
# F8 = 'instant_bookable'

# l_df1 = l_df[[F1, F2, F3, F4, F5, F6, F7, F8, 'price', 'reviews_per_month'
#               ]][(l_df[F1].notna())
#                  & (l_df[F2].notna())
#                  & (l_df[F3].notna())
#                  & (l_df[F4].notna())
#                  & (l_df[F5].notna())
#                  & (l_df[F6].notna())
#                  & (l_df[F7].notna())
#                  & (l_df[F8].notna())
#                  & (l_df['price'].replace('[\$,)]', '', regex=True).replace(
#                      '[(]', '-', regex=True).astype(float) < 300)
#                  & (l_df['reviews_per_month'].notna())]
# l_df1['price'] = l_df['price'].replace('[\$,)]', '', regex=True).replace(
#     '[(]', '-', regex=True).astype(float)
# l_df1.describe()


# In[15]:


price_lo_bound = 20
price_hi_bound = 500

l_df1 = l_df
l_df1[[
    'price', 'weekly_price', 'monthly_price', 'security_deposit',
    'cleaning_fee', 'extra_people'
]] = l_df[[
    'price', 'weekly_price', 'monthly_price', 'security_deposit',
    'cleaning_fee', 'extra_people'
]].replace('[\$,)]', '', regex=True).replace('[(]', '-',
                                             regex=True).astype(float)
l_df1['host_response_rate'] = l_df['host_response_rate'].replace(
    '[,\%)]', '', regex=True).replace('[(]', '-', regex=True).astype(float)
df_new = l_df1
df_new = df_new[
    (df_new['price'] >= price_lo_bound)
    & (df_new['price'] <= price_hi_bound)].drop(
        [
            'scrape_id',
            'listing_url',
            'thumbnail_url',
            'medium_url',
            'xl_picture_url',
            'last_scraped',
            'name',
            'host_id',
            'summary',
            'space',
            'description',
            'neighborhood_overview',
            'notes',
            'transit',
            'access',
            'interaction',
            'picture_url',
            'experiences_offered',
            'house_rules',
            'host_url',
            'host_name',
            'host_since',
            'host_location',
            'host_about',
            'neighborhood_overview',
            'host_thumbnail_url',
            'host_picture_url',
            'host_verifications',
            'host_acceptance_rate',
            'host_listings_count',
            'host_total_listings_count',
            'neighbourhood_group_cleansed',
            'amenities',
            'calendar_last_scraped',
            # these might need to be considered
            'first_review',
            'last_review',
            'number_of_reviews',
            'number_of_reviews_ltm',
            'review_scores_rating',
            #
            'license',
            'jurisdiction_names',
            #
            'latitude',
            'longitude',
            'host_neighbourhood',
            'calendar_updated',
            #                         'neighbourhood_cleansed',  # not sure
            'zipcode',
            'calculated_host_listings_count',
            'review_scores_value',
            'review_scores_cleanliness',
            'review_scores_location',
            'review_scores_accuracy',
            'review_scores_communication',
            'maximum_minimum_nights',
            'minimum_minimum_nights',
            'maximum_maximum_nights',
            'minimum_maximum_nights',
            'country',
            'host_response_rate',
            'host_is_superhost',
            'host_identity_verified',
            'street',
            'neighbourhood',
            'city',
            'state',
            'market',
            'smart_location',
            'country_code',
            'square_feet',
            'minimum_nights_avg_ntm',
            'maximum_nights_avg_ntm',
            'has_availability',
            'availability_30',
            'availability_60',
            'availability_90',
            'availability_365',
            'review_scores_checkin',
            'requires_license',
            'is_business_travel_ready',
            'calculated_host_listings_count_entire_homes',
            'calculated_host_listings_count_private_rooms',
            'calculated_host_listings_count_shared_rooms',
            'host_has_profile_pic',
            'require_guest_profile_picture',
            'require_guest_phone_verification',
            'is_location_exact',
            'bed_type',
            'beds',
            'bedrooms',
            'bathrooms',
            'cancellation_policy',
            'instant_bookable',
            'property_type',
            'room_type',
            'guests_included',
            'host_response_time'
        ],
        axis=1)  #.sort_values(by=['price'])


# In[16]:


df_new = df_new[df_new['reviews_per_month'].notna()]
# df_new['host_response_time'] = df_new['host_response_time'].fillna('NA')
# df_new['bathrooms'] = df_new['bathrooms'].fillna(df_new['bathrooms'].value_counts().index[0])
# df_new['bedrooms'] = df_new['bedrooms'].fillna(df_new['bedrooms'].value_counts().index[0])
# df_new['beds'] = df_new['beds'].fillna(df_new['beds'].value_counts().index[0])
df_new['weekly_price'] = df_new['weekly_price'].fillna(df_new['price']*7)
df_new['monthly_price'] = df_new['monthly_price'].fillna(df_new['price']*30)
df_new['security_deposit'] = df_new['security_deposit'].fillna(df_new['price']*2.75)
df_new['cleaning_fee'] = df_new['cleaning_fee'].fillna(df_new['price']*0.67)
df_new = df_new[df_new['minimum_nights'] <= 365]


# In[17]:


def occupancy_rate_est(data, alos, rr, coor):
    # alos: average length of stay
    # rr: review rate
    # coor: cap of occupancy rate

    data.loc[(data['minimum_nights'] <= alos) &
             (data['maximum_nights'] >= alos
              ), 'occupancy_%'] = data['reviews_per_month'] * alos / rr / 30

    data.loc[(data['maximum_nights'] <= alos), 'occupancy_%'] = data[
        'reviews_per_month'] * data['maximum_nights'] / rr / 30

    data.loc[(data['minimum_nights'] >= alos), 'occupancy_%'] = data[
        'reviews_per_month'] * data['minimum_nights'] / rr / 30

    data['occupancy_%'] = data['occupancy_%'].where(
        data['occupancy_%'] <= coor, coor)

    return data.head(10)


# In[18]:


occupancy_rate_est(df_new, 5.5, 0.72, 1)


# In[19]:


df_new['revenue'] = df_new['price']*df_new['occupancy_%']


# In[20]:


df_new['neighbourhood_cleansed'].value_counts().plot.bar()


# In[21]:


class MultiColumnLabelEncoder(LabelEncoder):
    """
    Wraps sklearn LabelEncoder functionality for use on multiple columns of a
    pandas dataframe.

    """
    def __init__(self, columns=None):
        self.columns = columns

    def fit(self, dframe):
        """
        Fit label encoder to pandas columns.

        Access individual column classes via indexig `self.all_classes_`

        Access individual column encoders via indexing
        `self.all_encoders_`
        """
        # if columns are provided, iterate through and get `classes_`
        if self.columns is not None:
            # ndarray to hold LabelEncoder().classes_ for each
            # column; should match the shape of specified `columns`
            self.all_classes_ = np.ndarray(shape=self.columns.shape,
                                           dtype=object)
            self.all_encoders_ = np.ndarray(shape=self.columns.shape,
                                            dtype=object)
            for idx, column in enumerate(self.columns):
                # fit LabelEncoder to get `classes_` for the column
                le = LabelEncoder()
                le.fit(dframe.loc[:, column].values)
                # append the `classes_` to our ndarray container
                self.all_classes_[idx] = (column,
                                          np.array(le.classes_.tolist(),
                                                   dtype=object))
                # append this column's encoder
                self.all_encoders_[idx] = le
        else:
            # no columns specified; assume all are to be encoded
            self.columns = dframe.iloc[:, :].columns
            self.all_classes_ = np.ndarray(shape=self.columns.shape,
                                           dtype=object)
            for idx, column in enumerate(self.columns):
                le = LabelEncoder()
                le.fit(dframe.loc[:, column].values)
                self.all_classes_[idx] = (column,
                                          np.array(le.classes_.tolist(),
                                                   dtype=object))
                self.all_encoders_[idx] = le
        return self

    def fit_transform(self, dframe):
        """
        Fit label encoder and return encoded labels.

        Access individual column classes via indexing
        `self.all_classes_`

        Access individual column encoders via indexing
        `self.all_encoders_`

        Access individual column encoded labels via indexing
        `self.all_labels_`
        """
        # if columns are provided, iterate through and get `classes_`
        if self.columns is not None:
            # ndarray to hold LabelEncoder().classes_ for each
            # column; should match the shape of specified `columns`
            self.all_classes_ = np.ndarray(shape=self.columns.shape,
                                           dtype=object)
            self.all_encoders_ = np.ndarray(shape=self.columns.shape,
                                            dtype=object)
            self.all_labels_ = np.ndarray(shape=self.columns.shape,
                                          dtype=object)
            for idx, column in enumerate(self.columns):
                # instantiate LabelEncoder
                le = LabelEncoder()
                # fit and transform labels in the column
                dframe.loc[:, column] =                    le.fit_transform(dframe.loc[:, column].values)
                # append the `classes_` to our ndarray container
                self.all_classes_[idx] = (column,
                                          np.array(le.classes_.tolist(),
                                                   dtype=object))
                self.all_encoders_[idx] = le
                self.all_labels_[idx] = le
        else:
            # no columns specified; assume all are to be encoded
            self.columns = dframe.iloc[:, :].columns
            self.all_classes_ = np.ndarray(shape=self.columns.shape,
                                           dtype=object)
            for idx, column in enumerate(self.columns):
                le = LabelEncoder()
                dframe.loc[:, column] = le.fit_transform(
                    dframe.loc[:, column].values)
                self.all_classes_[idx] = (column,
                                          np.array(le.classes_.tolist(),
                                                   dtype=object))
                self.all_encoders_[idx] = le
        return dframe

    def transform(self, dframe):
        """
        Transform labels to normalized encoding.
        """
        if self.columns is not None:
            for idx, column in enumerate(self.columns):
                dframe.loc[:, column] = self.all_encoders_[idx].transform(
                    dframe.loc[:, column].values)
        else:
            self.columns = dframe.iloc[:, :].columns
            for idx, column in enumerate(self.columns):
                dframe.loc[:, column] = self.all_encoders_[idx]                    .transform(dframe.loc[:, column].values)
        return dframe.loc[:, self.columns].values

    def inverse_transform(self, dframe):
        """
        Transform labels back to original encoding.
        """
        if self.columns is not None:
            for idx, column in enumerate(self.columns):
                dframe.loc[:, column] = self.all_encoders_[idx]                    .inverse_transform(dframe.loc[:, column].values)
        else:
            self.columns = dframe.iloc[:, :].columns
            for idx, column in enumerate(self.columns):
                dframe.loc[:, column] = self.all_encoders_[idx]                    .inverse_transform(dframe.loc[:, column].values)
        return dframe


# In[22]:


# find object columns
object_columns = df_new.iloc[:, :].select_dtypes(include=['object']).columns

# Multi-column label encoding
mcle = MultiColumnLabelEncoder(columns = object_columns)
df_go = mcle.fit_transform(df_new)


# In[23]:


### Split Data
# Randomly split dataframe into train and test with a certain ratio
msk = np.random.rand(len(df_go)) < 0.8

train = df_go[msk]
train1 = train.copy()
train1['price'] = train1['price'] * 2
train1['occupancy_%'] = train1['occupancy_%'] * 0
train1['revenue'] = train1['price'] * train1['occupancy_%']
train2 = train.copy()
train2['price'] = train1['price'] * 0
train2['occupancy_%'] = 1
train2['revenue'] = train2['price'] * train2['occupancy_%']
train = train.append(train1, ignore_index=True)
train = train.append(train2, ignore_index=True)

test = df_go[~msk]
test = test[test['price'] > 0]
test = test.drop_duplicates(subset='id', keep='first', inplace=False);


# In[24]:


from sklearn.utils import shuffle
train = shuffle(train)
test = shuffle(test)


# In[25]:


# Assign independent variables as x and the target variable as y
x_train = train.drop(['id', 'reviews_per_month','occupancy_%', 'revenue'], axis=1)
y_train = train['occupancy_%']
x_test = test.drop(['id', 'reviews_per_month','occupancy_%', 'revenue'], axis=1)
y_test = test['occupancy_%']


# In[26]:


# ### Feature Scaling
# # Normalize the range of independent variables to ensure the variable weights are not influenced by the scale difference as well as to improve the convergence speed of algorithms
# min_max_scaler = MinMaxScaler()
# x_train = min_max_scaler.fit_transform(x_train)
# x_test = min_max_scaler.fit_transform(x_test)


# ### Machine Learning Setups and Evaluations
# #### LightGBM

# In[61]:


# Grid Search
lgb = LGBMRegressor()
n_estimators = [5,10,20,30,40,50,100,150,200,250,300,350,400,450,500]
max_depth = [3,5,10,15]
param_grid = dict(max_depth=max_depth, n_estimators=n_estimators)
# kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=7)
grid_search = GridSearchCV(lgb, param_grid, n_jobs=-1, scoring='r2', cv=5, verbose=1)
grid_result = grid_search.fit(x_train, y_train)
# summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
# for mean, stdev, param in zip(means, stds, params):
#     print("%f (%f) with: %r" % (mean, stdev, param))


# In[62]:


# Plot grid search results
plt.figure(figsize=(12,6));
scores = np.array(means).reshape(len(max_depth), len(n_estimators))
stdevs = np.array(stds).reshape(len(max_depth), len(n_estimators))
for i, value in enumerate(max_depth):
    plt.plot(n_estimators, scores[i], label='max_depth = ' + str(value));
    plt.fill_between(n_estimators, scores[i]-stdevs[i], scores[i]+stdevs[i], color='#888888', alpha=0.4);
plt.xlim(0,500)
plt.ylim(0.75,1)
plt.legend()
plt.xlabel('n_estimators');
plt.ylabel('mean of R-square');
plt.savefig('n_estimators_vs_max_depth.png');


# In[320]:


# Set general hyperparameters for all ML packages using the grid search result
max_depth = 12
n_estimator = 1200
random_state = 110


# In[321]:


get_ipython().run_cell_magic('time', '', '# start = time.time()\nlgb = LGBMRegressor(max_depth=max_depth,\n                    n_estimators=n_estimator,\n                    random_state=random_state)\nlgb.fit(x_train, y_train, categorical_feature=list(object_columns.values))\n# code = m2c.export_to_python(lgb); # try next line first\nlgb.booster_.save_model(\'lgb_regressor.txt\')\n# from sklearn.externals import joblib\n# joblib.dump(lgb, \'lgb.pkl\')\n# acc_lgb = accuracy_score(y_test, lgb.predict(x_test))\n\ny_pred_lgb = lgb.predict(x_test)\n\n# fpr_lgb, tpr_lgb, _ = roc_curve(y_test, y_pred_lgb, drop_intermediate=False)\n# auc_lgb = roc_auc_score(y_test, y_pred_lgb)\n# end = time.time()\n# time_lgb = end - start\n# print("The entire evaluation takes", time_lgb, "s")')


# In[322]:


# #Cross Validation
# dftrainLGB = lgbDs(data = x_train, label = y_train, feature_name = list(x_train))

# params = {'objective': 'regression', 'max_depth': 30}
# cv_results = lgbcv(
#         params,
#         dftrainLGB,
#         num_boost_round=1100,
#         nfold=5,
#         metrics='rmse',
#         early_stopping_rounds=10,
#         categorical_feature=list(object_columns.values),

#         # This is what I added
#         stratified=False
#         )


# In[323]:


# plt.plot(list(range(1,1101,1)), cv_results['rmse-mean'])


# In[324]:


# ttt = Booster(model_file='lgb_regressor.txt')
# ttt.predict(x_test)
# gbm_pickle = joblib.load('lgb.pkl')


# In[325]:


get_ipython().run_cell_magic('time', '', "f, axs = plt.subplots(1, 3, figsize=(16,4))\nf.subplots_adjust(wspace=0.4);\n\naxs[0].plot(y_pred_lgb, y_test, 'gs', markersize=2)\naxs[0].set(xlim=(0, 1), ylim=(0, 1), xlabel='Pred. Occupancy%', ylabel='Occupancy%')\naxs[0].plot([0, 1000], [0, 1000], 'k--')\n\naxs[1].plot(x_test['price']*y_pred_lgb, x_test['price']*y_test, 'gs', markersize=2)\naxs[1].set(xlim=(0, 500), ylim=(0, 500), xlabel='Pred. Revenue', ylabel='Revenue')\naxs[1].plot([0, 500], [0, 500], 'k--')\n\naxs[2].plot(x_test['price']*y_pred_lgb, (x_test['price']*y_test - x_test['price']*y_pred_lgb), 'gs', markersize=2)\naxs[2].set(xlim=(0, 200), ylim=(-200, 200), xlabel='Pred. Occupancy%', ylabel='Occupancy Residual')\n\n\nprint('RMSE =', np.sqrt(mean_squared_error(y_test, y_pred_lgb)))\nprint('R^2 =', r2_score(y_test, y_pred_lgb))\nprint('RMSE =', np.sqrt(mean_squared_error(x_test['price']*y_test, x_test['price']*y_pred_lgb)))\nprint('R^2 =',r2_score(x_test['price']*y_test, x_test['price']*y_pred_lgb))")


# In[29]:


plt.figure(figsize=(6, 6));
pd.Series(lgb.feature_importances_,x_train.columns).sort_values(ascending=True).plot.barh(width=0.8);
plt.title('Feature Importance in Gradient Boosting (LGBM)');
plt.show();


# In[30]:


## heatmap
# f,ax = plt.subplots(figsize=(8, 8))
# sns.set(font_scale=1.0)
# sns.heatmap(df_go.corr(), annot=True, linewidths=1, cmap = plt.cm.RdYlBu_r, vmin = -0.5, vmax = 0.8,fmt= '.1f')
# plt.title('Correlation Heatmap')
# plt.show()
# sns.reset_orig()


# In[350]:


test_ID = 1111

max_price = int(x_test['price'].iloc[test_ID]*2)
price_range = range(0, max_price, 1)

x_in = pd.DataFrame(np.repeat(x_test.iloc[[test_ID]].values, max_price, axis=0))
x_in.columns = x_test.columns
x_in['price'] = pd.DataFrame(list(price_range)).iloc[:, 0]

pred_occupancy = lgb.predict(x_in)
pred_occupancy[pred_occupancy > 1] = 1 # ensure that occupancy% does not go above 100%
pred_occupancy[pred_occupancy < 0] = 0 # ensure that occupancy% does not go below 0%
pred_revenue = pred_occupancy*x_in['price']

# -------------------------------------------------------------
z_o = np.polyfit(price_range, pred_occupancy, 5)
p_o = np.poly1d(z_o)
xp_o = np.linspace(0, max_price, max_price)

z_r = np.polyfit(price_range, pred_revenue*30, 4)
p_r = np.poly1d(z_r)
xp_r = np.linspace(0, max_price, max_price)
# -------------------------------------------------------------
print('original price is $', x_test['price'].iloc[test_ID])
print('original monthly revenue is $', x_test['price'].iloc[test_ID]*y_test.iloc[test_ID]*30)
print('model predicted monthly revenue is $', x_test['price'].iloc[test_ID]*y_pred_lgb[test_ID]*30)
print('optimal monthly revenue is $', pred_revenue.max()*30)
print('optimized price is $', np.argmax(pred_revenue))

f,axs=plt.subplots(1,2, figsize=(20,10))

axs[0].plot(price_range, pred_occupancy, 'gs',  markersize = 2, label='predicted results');
axs[0].plot(xp_o,p_o(xp_o), linewidth=3, linestyle='--', label='polyfit')
axs[0].set(xlim = (0, max_price), ylim = (-0.1,1));
axs[0].set_xlabel('price', fontsize=28)
axs[0].set_ylabel('pred. occupancy%', fontsize=28)
axs[0].tick_params(axis="x", labelsize=24)
axs[0].tick_params(axis="y", labelsize=24)

axs[1].plot(price_range, pred_revenue*30, 'gs', markersize = 2);
axs[1].plot(xp_r,p_r(xp_r), linewidth=3, linestyle='--')
axs[1].set(xlim = (0, max_price), xlabel = 'price', ylabel = 'revenue');
axs[1].set_xlabel('price', fontsize=28)
axs[1].set_ylabel('pred. revenue', fontsize=28)
axs[1].tick_params(axis="x", labelsize=24)
axs[1].tick_params(axis="y", labelsize=24)

# adjust the border width
for axis in ['top','bottom','left','right']:
    axs[0].spines[axis].set_linewidth(2)
    axs[1].spines[axis].set_linewidth(2)

f.subplots_adjust(wspace=0.4);


# In[32]:


x_test.iloc[497]


# In[199]:


pred_revenue


# In[32]:


get_ipython().run_cell_magic('time', '', "# create final table\ndf_out = pd.DataFrame(columns=['o_price', 'o_mon_rev', 'pred_mon_rev', 'opt_price', 'opt_mon_rev'])\n\nfor test_ID in list(range(len(x_test))):\n    max_price = int(x_test['price'].iloc[test_ID]*2)   \n    price_range = range(0, max_price, 1)\n\n    x_in = pd.DataFrame(np.repeat(x_test.iloc[[test_ID]].values, max_price, axis=0))\n    x_in.columns = x_test.columns\n    x_in['price'] = pd.DataFrame(list(price_range)).iloc[:, 0]\n\n    pred_occupancy = lgb.predict(x_in)\n    pred_occupancy[pred_occupancy > 1] = 1 # ensure that occupancy% does not go above 100%\n    pred_occupancy[pred_occupancy < 0] = 0 # ensure that occupancy% does not go below 0%\n    pred_revenue = pred_occupancy*x_in['price']\n\n    df_out = df_out.append({'o_price':x_test['price'].iloc[test_ID], 'o_mon_rev':x_test['price'].iloc[test_ID]*y_test.iloc[test_ID]*30, 'pred_mon_rev':x_test['price'].iloc[test_ID]*y_pred_lgb[test_ID]*30, 'opt_price':np.argmax(pred_revenue), 'opt_mon_rev':pred_revenue.max()*30}, ignore_index=True)")


# In[33]:


df_out['diff_mon_rev'] = df_out['opt_mon_rev'] - df_out['o_mon_rev']
df_out['diff_mon_rev%'] = df_out['diff_mon_rev']/df_out['o_mon_rev']*100

# df_out = df_out[(df_out['o_price'] > 0) & (df_out['o_mon_rev'] > 0)]
df_out.describe()
print(round(df_out['diff_mon_rev%'][df_out['diff_mon_rev%'] > 0].count()/df_out.shape[0], 4)*100, '% of users benifit from the tool')


# In[200]:


df_ben = df_out[df_out['diff_mon_rev'] > 0]
print('For those who benifit')
print('gross revenue increase is', round((df_ben['diff_mon_rev'].sum()/df_ben['o_mon_rev'].sum())*100, 2), '%')

print('25% of these users get', round(df_ben.describe().iloc[6,6], 2), '%+ monthly revenue increase')
print('50% of these users get', round(df_ben.describe().iloc[5,6], 2), '%+ monthly revenue increase')
print('75% of these users get', round(df_ben.describe().iloc[4,6], 2), '%+ monthly revenue increase')


# #### Model (web domain)

# In[739]:


get_ipython().run_cell_magic('time', '', "x_go = df_go.drop(['id', 'reviews_per_month','occupancy_%', 'revenue'], axis=1)\ny_go = df_go['occupancy_%']\n\nlgb = LGBMRegressor(max_depth=max_depth,\n                    n_estimators=n_estimator,\n                    random_state=random_state)\nlgb.fit(x_go, y_go, categorical_feature=list(object_columns.values))\nlgb.booster_.save_model('lgb_regressor.txt')")


# In[796]:


df_new['guests_included'].max()


# In[760]:


mcle.all_classes_


# In[798]:


a_list = ['flexible', 'moderate', 'strict', 'strict_14_with_grace_period',
       'super_strict_30', 'super_strict_60']

for i, s in enumerate(a_list):
    print('<option value="%d">%s</option>' %(i, s))


# In[777]:


df_new['neighbourhood_cleansed'].head()


# #### RandomForest

# In[384]:


# Set general hyperparameters for all ML packages using the grid search result
max_depth = 10
n_estimator = 120
random_state = 110


# In[386]:


get_ipython().run_cell_magic('time', '', '# start = time.time()\nrf = RandomForestRegressor(max_depth=max_depth,\n                    n_estimators=n_estimator,\n                    random_state=random_state)\nrf.fit(x_train, y_train)\n# acc_rf = accuracy_score(y_test, rf.predict(x_test))\ny_pred_rf = rf.predict(x_test)\n# fpr_rf, tpr_rf, _ = roc_curve(y_test, y_pred_rf, drop_intermediate=False)\n# auc_rf = roc_auc_score(y_test, y_pred_rf)\n# end = time.time()\n# time_rf = end - start\n# print("The entire evaluation takes", time_rf, "s")')


# In[387]:


get_ipython().run_cell_magic('time', '', "f, axs = plt.subplots(1, 2)\n\naxs[0].plot(y_test, y_pred_rf, 'gs', markersize=3)\naxs[0].set(xlim=(0, 1), ylim=(0, 1), xlabel='Est. Occupancy%', ylabel='Pred. Occupancy%', title= 'Gradient Boosting (RF)')\naxs[0].plot([0, 1000], [0, 1000], 'k--')\n\naxs[1].plot(x_test['price']*y_test, x_test['price']*y_pred_rf, 'gs', markersize=3)\naxs[1].set(xlim=(0, 500), ylim=(0, 500), xlabel='Est. Revenue', ylabel='Pred. Revenue', title= 'Gradient Boosting (RF)')\naxs[1].plot([0, 500], [0, 500], 'k--')\nf.subplots_adjust(wspace=0.4);\n\nprint('RMSE =', np.sqrt(mean_squared_error(y_test, y_pred_rf)))\nprint('R^2 =', r2_score(y_test, y_pred_rf))\nprint('RMSE =', np.sqrt(mean_squared_error(x_test['price']*y_test, x_test['price']*y_pred_rf)))\nprint('R^2 =',r2_score(x_test['price']*y_test, x_test['price']*y_pred_rf))")


# In[305]:


test_ID = 1428
if x_test['price'].iloc[test_ID] > 0:
    max_price = int(x_test['price'].iloc[test_ID]*1.5)
else:
    max_price = 500

price_range = range(0, max_price, 1)

x_in = pd.DataFrame(np.repeat(x_test.iloc[[test_ID]].values, max_price, axis=0))
x_in.columns = x_test.columns
x_in['price'] = pd.DataFrame(list(price_range)).iloc[:, 0]

pred_occupancy = rf.predict(x_in)
pred_occupancy[pred_occupancy < 0] = 0 # ensure that occupancy% does not go below 0%
pred_revenue = pred_occupancy*x_in['price']

xnew1 = np.linspace(0, max_price-2, 30)
spl1 = make_interp_spline(price_range, pred_revenue, k=3)  #BSpline object
power_smooth1 = spl1(xnew1)

xnew2 = np.linspace(0, max_price-2, 30)
spl2 = make_interp_spline(price_range, pred_occupancy, k=3)  #BSpline object
power_smooth2 = spl2(xnew2)

print('original price is $', x_test['price'].iloc[test_ID])
print('original monthly revenue is $', x_test['price'].iloc[test_ID]*y_test.iloc[test_ID]*30)
print('model predicted monthly revenue is $', x_test['price'].iloc[test_ID]*y_pred_rf[test_ID]*30)
print('optimal monthly revenue is $', pred_revenue.max()*30)
print('optimized price is $', np.argmax(pred_revenue))

f,axs=plt.subplots(1,2)

axs[0].plot(price_range, pred_occupancy, 'gs', markersize = 1);
axs[0].plot(xnew2,power_smooth2)
axs[0].set(xlim = (0, max_price), ylim = (0,1) , xlabel = 'price', ylabel = 'occupancy %');

axs[1].plot(price_range, pred_revenue, 'gs', markersize = 1);
axs[1].plot(xnew1,power_smooth1)
axs[1].set(xlim = (0, max_price), xlabel = 'price', ylabel = 'revenue');

f.subplots_adjust(wspace=0.4);


# In[ ]:




