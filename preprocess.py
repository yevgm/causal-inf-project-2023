import pandas as pd


def preprocess_data_waged(raw_data):
    # Education attainment: A_HGA
    # < 41 is under college
    # 42 college
    # 43 Bachelors
    # 44 Masters
    # 45 Professional degree
    # 46 Doctorate
    edu_dict = {
        41: 'U_college',
        42: 'college',
        43: 'bachelors',
        44: 'masters',
        45: 'prof',
        46: 'doctorate'
    }

    def encode_edu(x):
        if x > 0 and x <= 41:
            x = edu_dict[41]
        elif x > 0 and x <= 46:
            x = edu_dict[x]
        else:
            x = None

        return x

    def prep_race(x):
        if x >= 5:
            x = 5
        return x

    # Age - A_AGE
    raw_data = raw_data[raw_data['A_AGE'] >= 18]

    # Education - A_HGA
    raw_data.loc[:, 'A_HGA'] = raw_data['A_HGA'].apply(encode_edu)
    assert raw_data.isna().sum().sum() == 0

    # Sex - A_SEX
    raw_data.loc[:, 'A_SEX'] = 1 * (raw_data['A_SEX'] == 1)

    # Create bins from age.
    # age_bins = np.arange(18, 120, 5)
    # labels = age_bins[1:]
    # raw_data.loc[:, 'A_AGE'] = pd.cut(raw_data['A_AGE'], age_bins, labels=labels)

    # Hourly rate: CALC_H_RATE
    # ERN_SRCE earn source is 1
    # and A_USLHRS - usual hours per week
    # and A_GRSWK payment per week

    raw_data = raw_data[(raw_data['ERN_SRCE'] == 1) & (raw_data['A_GRSWK'] > 0) & (raw_data['A_USLHRS'] > 0)]
    raw_data['CALC_H_RATE'] = raw_data['A_GRSWK'] / raw_data['A_USLHRS']

    # Industry: A_MJIND
    raw_data = raw_data[(raw_data['A_MJIND'] > 0) & (raw_data['A_MJIND'] < 14)]

    # Race: PRDTRACE
    raw_data.loc[:, 'PRDTRACE'] = raw_data['PRDTRACE'].apply(prep_race)

    # Create new subset of chosen columns
    data = raw_data[['A_HGA', 'PRDTRACE', 'CALC_H_RATE', 'A_SEX', 'A_AGE']]  # 'A_MJIND'
    data = data.rename(columns={'A_HGA': 'T', 'CALC_H_RATE': 'Y'})

    # Make the data compatible for covariate adjustment - make categorical variables into binary ones
    processed_data = pd.get_dummies(data, columns=['T', 'PRDTRACE'])  # 'A_MJIND'

    treatments = []
    for ed in edu_dict.values():
        treatments.append(processed_data['T_' + ed])
        processed_data = processed_data.drop(columns=['T_' + ed])

    data_subsets = []
    for t_ind, val in enumerate(edu_dict.values()):
        if val != 'U_college':
            control_treatment = treatments[0]
            positive_treatment = treatments[t_ind]
            curr_treatment = (1 - control_treatment) + positive_treatment
            curr_treatment[curr_treatment == 1] = None
            curr_treatment[curr_treatment == 2] = 1
            curr_treatment.name = 'T'

            X = pd.concat((processed_data, curr_treatment), axis=1).dropna()
            y = X['Y']
            X = X.drop(columns=['Y'])
            data_subsets.append((X, y, val))
            print(val, X['T'].sum())

    return data_subsets, data


def preprocess_data_selfemp(raw_data):
    # Education attainment: A_HGA
    # < 41 is under college
    # 42 college
    # 43 Bachelors
    # 44 Masters
    # 45 Professional degree
    # 46 Doctorate
    edu_dict = {
        41: 'U_college',
        42: 'college',
        43: 'bachelors',
        44: 'masters',
        45: 'prof',
        46: 'doctorate'
    }

    def encode_edu(x):
        if x > 0 and x <= 41:
            x = edu_dict[41]
        elif x > 0 and x <= 46:
            x = edu_dict[x]
        else:
            x = None

        return x

    def prep_race(x):
        if x >= 5:
            x = 5
        return x

    # Age - A_AGE
    raw_data = raw_data[raw_data['A_AGE'] >= 18]

    # Education - A_HGA
    raw_data.loc[:, 'A_HGA'] = raw_data['A_HGA'].apply(encode_edu)
    assert raw_data.isna().sum().sum() == 0

    # Sex - A_SEX
    raw_data.loc[:, 'A_SEX'] = 1 * (raw_data['A_SEX'] == 1)

    # Create bins from age.
    # age_bins = np.arange(18, 120, 5)
    # labels = age_bins[1:]
    # raw_data.loc[:, 'A_AGE'] = pd.cut(raw_data['A_AGE'], age_bins, labels=labels)

    # Total earnings: PEARNVAL
    # ERN_SRCE earn source is 2 (self emp)
    # and ERN_OTR - no other salary other than self employment
    # and A_CLSWKR - one more field for self employment

    raw_data = raw_data[(raw_data['ERN_SRCE'] == 2) & (raw_data['ERN_OTR'] == 2) & (raw_data['PEARNVAL'] > 0) & (
                raw_data['A_CLSWKR'] >= 5) & (raw_data['A_CLSWKR'] <= 6)]

    # Industry: A_MJIND
    # raw_data = raw_data[(raw_data['A_MJIND'] > 0) & (raw_data['A_MJIND'] < 14)]

    # Race: PRDTRACE
    raw_data.loc[:, 'PRDTRACE'] = raw_data['PRDTRACE'].apply(prep_race)

    # Create new subset of chosen columns
    data = raw_data[['A_HGA', 'PRDTRACE', 'PEARNVAL', 'A_SEX', 'A_AGE']]
    data = data.rename(columns={'A_HGA': 'T', 'PEARNVAL': 'Y'})

    # Make the data compatible for covariate adjustment - make categorical variables into binary ones
    processed_data = pd.get_dummies(data, columns=['T', 'PRDTRACE'])
    # processed_data = pd.get_dummies(data, columns=['T'])

    treatments = []
    for ed in edu_dict.values():
        treatments.append(processed_data['T_' + ed])
        processed_data = processed_data.drop(columns=['T_' + ed])

    # y = processed_data['Y']
    # processed_data = processed_data.drop(columns=['Y'])

    data_subsets = []
    for t_ind, val in enumerate(edu_dict.values()):
        if val != 'U_college':
            control_treatment = treatments[0]
            positive_treatment = treatments[t_ind]
            curr_treatment = (1 - control_treatment) + positive_treatment
            curr_treatment[curr_treatment == 1] = None
            curr_treatment[curr_treatment == 2] = 1
            curr_treatment.name = 'T'

            X = pd.concat((processed_data, curr_treatment), axis=1).dropna()
            y = X['Y']
            X = X.drop(columns=['Y'])
            data_subsets.append((X, y, val))
            print(val, X['T'].sum())

    return data_subsets, data


if __name__ == '__main__':
    raw_data = pd.read_csv('./data/pppub22.csv')
    data_subsets, data = preprocess_data_selfemp(raw_data)