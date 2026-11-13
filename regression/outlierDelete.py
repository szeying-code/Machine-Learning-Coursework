import pandas as pd

def outlierDelete(all_df):

    feature_cols = [c for c in all_df.columns if c not in ['pCR (outcome)', 'RelapseFreeSurvival (outcome)',
                                                                     'ER', 'PgR', 'HER2', 'TrippleNegative',
                                                                     'ChemoGrade', 'Proliferation', 'HistologyType',
                                                                     'LNStatus', 'TumourStage', 'Gene']]

    outlier_idx = set()
    for col in feature_cols:
        Q1 = all_df[col].quantile(0.25)
        Q3 = all_df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5*IQR
        upper = Q3 + 1.5*IQR
        outlier_idx.update(all_df[(all_df[col] < lower) | (all_df[col] > upper)].index)

    #Delete all abnormal lines
    return all_df.drop(outlier_idx)
