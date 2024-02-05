import time
import numpy as np
import pandas as pd
import seaborn as sns
import lightgbm as lgb
import warnings
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Qt5Agg')

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
warnings.filterwarnings('ignore')

df = pd.read_csv("Datasets/iyzico_data.csv", parse_dates=["transaction_date"])
df.drop("Unnamed: 0", axis=1, inplace=True)
df.head()
df.shape

#İlk ve Son İşlem Tarihleri:
df["transaction_date"].min() #2018-01-01
df["transaction_date"].max() #2020-12-31

#Her üye iş yerindeki toplam işlem sayısı kaçtır?
df.groupby(["merchant_id"])["Total_Transaction"].sum()

#Her üye iş yerindeki toplam ödeme miktarı kaçtır?
df.groupby(["merchant_id"]).agg({"Total_Paid": "sum"})

#üye iş yerlerinin her bir yıl içerisindeki transaction count grafiklerini gözlemleyiniz.
for id in df.merchant_id.unique():
    plt.figure(figsize=(15,25))
    plt.subplot(3, 1, 1, title=str(id) + ' 2018-2019 Transaction Count')
    df[(df.merchant_id == id) & (df.transaction_date >= "2018-01-01") & (df.transaction_date < "2019-01-01")][
        "Total_Transaction"].plot()
    plt.xlabel('')
    plt.subplot(3, 1, 2, title=str(id) + ' 2019-2020 Transaction Count')
    df[(df.merchant_id == id) & (df.transaction_date >= "2019-01-01") & (df.transaction_date < "2020-01-01")][
        "Total_Transaction"].plot()
    plt.xlabel('')
    plt.show()

#Feature Engineering

# Date Features

def create_date_features(df, date_column):
    df['month'] = df[date_column].dt.month
    df['day_of_month'] = df[date_column].dt.day
    df['day_of_year'] = df[date_column].dt.dayofyear
    df['day_of_week'] = df[date_column].dt.dayofweek
    df['year'] = df[date_column].dt.year
    df["is_wknd"] = df[date_column].dt.weekday // 4
    df['is_month_start'] =df[date_column].dt.is_month_start.astype(int)
    df['is_month_end'] = df[date_column].dt.is_month_end.astype(int)
    df['quarter'] = df[date_column].dt.quarter
    df['is_quarter_start'] = df[date_column].dt.is_quarter_start.astype(int)
    df['is_quarter_end'] = df[date_column].dt.is_quarter_end.astype(int)
    df['is_year_start'] = df[date_column].dt.is_year_start.astype(int)
    df['is_year_end'] = df[date_column].dt.is_year_end.astype(int)
    return df

df = create_date_features(df, "transaction_date")

# Üye iş yerlerinin yıl ve ay bazında işlem sayılarının incelenmesi
df.groupby(["merchant_id", "year", "month"]).agg({"Total_Transaction": "sum"})

# Üye iş yerlerinin yıl ve ay bazında toplam ödeme miktarlarının incelenmesi
df.groupby(["merchant_id", "year", "month"]).agg({"Total_Paid": "sum"})

# Lag/Shifted Features

def random_noise(dataframe):
    return np.random.normal(scale=1.6, size=(len(dataframe),))

def lag_features(dataframe, lags):
    for lag in lags:
        dataframe['sales_lag_' + str(lag)] = dataframe.groupby(["merchant_id"])['Total_Transaction'].transform(
            lambda x: x.shift(lag)) + random_noise(dataframe)
    return dataframe

df = lag_features(df, [91, 98, 105, 112, 119, 126, 135, 150, 182, 364, 450, 546, 650, 720])

#Rolling Mean Features
def roll_mean_features(dataframe, windows):
    for window in windows:
        dataframe['sales_roll_mean_' + str(window)] = dataframe.groupby(["merchant_id"])['Total_Transaction']. \
                                                          transform(
            lambda x: x.shift(1).rolling(window=window, min_periods=10, win_type="triang").mean()) + random_noise(
            dataframe)
    return dataframe


df = roll_mean_features(df, [91,92,178,179,180,181,182,359,360,361,449,450,451,539,540,541,629,630,631,720])
#Exponentially Weighted Mean Features
def ewm_features(dataframe, alphas, lags):
    for alpha in alphas:
        for lag in lags:
            dataframe['sales_ewm_alpha_' + str(alpha).replace(".", "") + "_lag_" + str(lag)] = \
                dataframe.groupby(["merchant_id"])['Total_Transaction'].transform(lambda x: x.shift(lag).ewm(alpha=alpha).mean())
    return dataframe

alphas = [0.95, 0.9, 0.8, 0.7, 0.5]
lags = [91, 98, 105, 112, 120, 135, 150, 165, 180, 270, 300, 365, 546, 720]

df = ewm_features(df, alphas, lags)

# Özellik Çıkarımı

df["is_black_friday"] = 0
df.loc[df["transaction_date"].isin(["2018-11-22","2018-11-23","2019-11-29","2019-11-30"]) ,"is_black_friday"] = 1


df["is_summer_solstice"] = 0
df.loc[df["transaction_date"].isin(["2018-06-19","2018-06-20","2018-06-21","2018-06-22",
                                    "2019-06-19","2019-06-20","2019-06-21","2019-06-22",]) ,"is_summer_solstice"]=1
# One-Hot Encoding


df = pd.get_dummies(df, columns=['merchant_id','day_of_week', 'month'])

#Custom Cost Function(müşteri maliyet fonksiyon)
def smape(preds, target):
    n = len(preds)
    masked_arr = ~((preds == 0) & (target == 0))
    preds, target = preds[masked_arr], target[masked_arr]
    num = np.abs(preds - target)
    denom = np.abs(preds) + np.abs(target)
    smape_val = (200 * np.sum(num / denom)) / n
    return smape_val
def lgbm_smape(preds, train_data):
    labels = train_data.get_label()
    smape_val = smape(np.expm1(preds), np.expm1(labels))
    return 'SMAPE', smape_val, False

#MODEL:
df['Total_Transaction'] = np.log1p(df["Total_Transaction"].values)
train = df.loc[(df["transaction_date"] < "2020-10-01"), :]
val = df.loc[(df["transaction_date"] >= "2020-10-01"), :]

cols = [col for col in train.columns if col not in ['transaction_date',"year", "Total_Transaction", "Total_Paid"]]

Y_train = train['Total_Transaction']
X_train = train[cols]

Y_val = val['Total_Transaction']
X_val = val[cols]

lgb_params = {'num_leaves': 10,
              'learning_rate': 0.02,
              'feature_fraction': 0.8,
              'max_depth':5,
              'verbose': 0,
              'num_boost_round': 10000,
              'early_stopping_rounds': 200,
              'nthread': -1}

lgbtrain =lgb.Dataset(data=X_train, label=Y_train, feature_name=cols)
lgbval = lgb.Dataset(data=X_val, label=Y_val, reference=lgbtrain, feature_name=cols)

model = lgb.train(lgb_params, lgbtrain,
                  valid_sets=[lgbtrain, lgbval],
                  num_boost_round=lgb_params['num_boost_round'],
                  callbacks=[lgb.early_stopping(stopping_rounds=200),lgb.log_evaluation(500)],
                  feval=lgbm_smape)
y_pred_val = model.predict(X_val, num_iteration=model.best_iteration)

smape(np.expm1(y_pred_val), np.expm1(Y_val))
#smape: 21.496199934849827

#Feature İmportance:
def plot_lgb_importances(model, plot=False, num=10):
    gain = model.feature_importance('gain')
    feat_imp = pd.DataFrame({'feature': model.feature_name(),
                             'split': model.feature_importance('split'),
                             'gain': 100 * gain / gain.sum()}).sort_values('gain', ascending=False)
    if plot:
        plt.figure(figsize=(10, 10))
        sns.set(font_scale=1)
        sns.barplot(x="gain", y="feature", data=feat_imp[0:25])
        plt.title('feature')
        plt.tight_layout()
        plt.show()
    else:
        print(feat_imp.head(num))
    return feat_imp
plot_lgb_importances(model, num=30, plot=True)

#FİNAL MODEL:
feat_imp = plot_lgb_importances(model, num=200)

importance_zero = feat_imp[feat_imp["gain"] == 0]["feature"].values

imp_feats = [col for col in cols if col not in importance_zero]

Y_train = train['Total_Transaction']
X_train = train[imp_feats]

Y_val = val['Total_Transaction']
X_val = val[imp_feats]

lgbtrain =lgb.Dataset(data=X_train, label=Y_train, feature_name=imp_feats)
lgbval = lgb.Dataset(data=X_val, label=Y_val, reference=lgbtrain, feature_name=imp_feats)

model = lgb.train(lgb_params, lgbtrain,
                  valid_sets=[lgbtrain, lgbval],
                  num_boost_round=lgb_params['num_boost_round'],
                  callbacks=[lgb.early_stopping(stopping_rounds=200),lgb.log_evaluation(500)],
                  feval=lgbm_smape)

y_pred_val = model.predict(X_val, num_iteration=model.best_iteration)

smape(np.expm1(y_pred_val), np.expm1(Y_val))
#smape:21.3694345792











