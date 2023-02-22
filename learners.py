from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import pandas as pd
import numpy as np
from tqdm.notebook import tqdm


def calc_s_learner_ate(model, X_in, num_bootstrap=100):
    # choose 60% from rhe dataset and calculate ate num_bootstrap times
    ate_list = []
    for it in range(num_bootstrap):
        subset = X_in.sample(frac=0.6)

        X = subset.copy().reset_index(drop=True)
        t0 = pd.Series(np.zeros((len(X),)))
        t1 = t0 + 1
        t0.name = 'T'
        t1.name = 'T'
        data0 = pd.concat((X, t0), axis=1)
        data1 = pd.concat((X, t1), axis=1)
        y0 = model.predict(data0)
        y1 = model.predict(data1)
        ate = (y1 - y0).mean()
        ate_list.append(ate)

    lb, med, up = np.quantile(ate_list, q=[0.25, 0.5, 0.75])
    med = np.around(med, decimals=2)
    lb = np.around(lb, decimals=2)
    up = np.around(up, decimals=2)
    return med, lb, up


def s_learner_att_ate_per_treatment(data_subsets, seed):
    # define models
    rf_cls = RandomForestRegressor(max_depth=10, random_state=seed)
    xgb_cls = GradientBoostingRegressor(max_depth=4, random_state=seed)
    linear = LinearRegression()
    models = [(linear, 'Linear'), (rf_cls, 'Random Forest'), (xgb_cls, 'XGBoost')]

    print('S-Learner results:')
    ate_data_dict, att_data_dict = {}, {}
    for idx, (model, model_name) in enumerate(tqdm(models)):
        ate_subset_out, att_subset_out = [], []
        row_names = []
        for data in data_subsets:
            X = data[0]
            y = data[1]
            name = data[2]
            model.fit(X, y)
            med, lb, up = calc_s_learner_ate(model, X.drop(columns=['T']))
            ate_subset_out.append((med, lb, up))

            treatment = X['T']
            X_in = X[X['T'] == 1].copy().drop(columns=['T'])
            med, lb, up = calc_s_learner_ate(model, X_in)
            att_subset_out.append((med, lb, up))

            row_names.append(name)

        ate_data_dict[model_name] = pd.Series(ate_subset_out, name=name)
        att_data_dict[model_name] = pd.Series(att_subset_out, name=name)

    ate_out = pd.DataFrame(ate_data_dict)
    ate_out.index = row_names

    att_out = pd.DataFrame(att_data_dict)
    att_out.index = row_names
    return ate_out, att_out


def calc_ipw_ate_att(model, X_in, t_in, y_in, num_bootstrap=100):
    eps = 1e-6
    # choose 60% from rhe dataset and calculate ate num_bootstrap times
    ipw_ate_list, ipw_att_list = [], []
    for it in range(num_bootstrap):
        rand_indices = np.random.randint(len(X_in), size=(int(len(X_in) * 0.6),))
        subset = X_in.iloc[rand_indices]
        X = subset.reset_index(drop=True)

        t = t_in.iloc[rand_indices].reset_index(drop=True)
        y = y_in.iloc[rand_indices].reset_index(drop=True)

        # predict propensity score
        propensity = model.predict_proba(X)

        # ipw_ate = ((t * y / (propensity[:, 1] + eps)) - (1 - t) * y / (1 - propensity[:, 1] + eps)).mean()
        ipw_ate = (t * y / propensity[:, 1]).sum() / (t / propensity[:, 1]).sum() - (
                    (1 - t) * y / (1 - propensity[:, 1] + eps)).sum() / (((1 - t) / (1 - propensity[:, 1] + eps)).sum())
        ipw_att = (t * y).sum() / t.sum() - ((1 - t) * y * propensity[:, 1] / (1 - propensity[:, 1] + eps)).sum() / (
            ((1 - t) * propensity[:, 1] / (1 - propensity[:, 1] + eps)).sum())
        ipw_ate_list.append(ipw_ate)
        ipw_att_list.append(ipw_att)

    lb, med, up = np.quantile(ipw_ate_list, q=[0.25, 0.5, 0.75])
    med_ate = np.around(med, decimals=2)
    lb_ate = np.around(lb, decimals=2)
    up_ate = np.around(up, decimals=2)

    lb, med, up = np.quantile(ipw_att_list, q=[0.25, 0.5, 0.75])
    med_att = np.around(med, decimals=2)
    lb_att = np.around(lb, decimals=2)
    up_att = np.around(up, decimals=2)
    return med_ate, lb_ate, up_ate, med_att, lb_att, up_att

def ipw_att_ate_per_treatment(data_subsets, seed):
    # define models
    rf_cls = RandomForestClassifier(max_depth=4, random_state=seed)
    xgb_cls = GradientBoostingClassifier(max_depth=4, random_state=seed)

    logistic = LogisticRegression(max_iter=10000, tol=0.1)
    scaler = StandardScaler()
    linear = Pipeline(steps=[("scaler", scaler), ("logistic", logistic)])

    models = [(linear, 'Linear'), (rf_cls, 'Random Forest'), (xgb_cls, 'XGBoost')]

    ate_data_dict, att_data_dict = {}, {}
    for idx, (model, model_name) in enumerate(tqdm(models)):
        ate_subset_out, att_subset_out = [], []
        row_names = []
        for data in data_subsets:
            t = data[0]['T']
            X = data[0].drop(columns=['T']).copy()
            y = data[1]
            name = data[2]
            model.fit(X, t)
            med_ate, lb_ate, up_ate, med_att, lb_att, up_att = calc_ipw_ate_att(model, X, t, y)
            ate_subset_out.append((med_ate, lb_ate, up_ate))
            att_subset_out.append((med_att, lb_att, up_att))

            row_names.append(name)

            # if model_name == 'XGBoost':
            #     plt.figure()
            #     propensity = model.predict_proba(X)[:, 1]
            #     prop0 = propensity[np.where(np.array(t) == 0)]
            #     prop1 = propensity[np.where(np.array(t) == 1)]
            #     plt.hist(prop0, bins=20, density=True, alpha=0.3)
            #     plt.hist(prop1, bins=20, density=True, alpha=0.3)

        ate_data_dict[model_name] = pd.Series(ate_subset_out, name=name)
        att_data_dict[model_name] = pd.Series(att_subset_out, name=name)

    return pd.DataFrame(ate_data_dict), pd.DataFrame(att_data_dict)