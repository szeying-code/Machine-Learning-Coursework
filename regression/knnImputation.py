from sklearn.neighbors import KNeighborsClassifier

def knnImputation(predict, all_df=None):#Incoming all_def
    """
    Since KNN performs poorly in high-dimensional Settings, a correlation analysis is first conducted between the features
    to be imputed and the remaining features without missing values.
    Select the two features with the greatest correlation, and then predict and replace 999 in all_df with KNN
    """

    #Calculate the correlation coefficients between the predict feature and the remaining features
    corrMatt = all_df.corr(numeric_only=True)
    predict_corr=corrMatt[predict].abs().sort_values(ascending=False)
    #print(predict_corr)
    #Select the two most relevant features
    top_cor_feature=predict_corr.index[1:3]
    cols=list(top_cor_feature)
    #print(cols)

    #Create columns for prediction
    cols.append(predict)

    #Create a dataframe for prediction
    predict_df = all_df.loc[:, cols]

    #Obtain the row sets with missing values and the row sets without missing values respectively
    predict_df_999 = predict_df[predict_df[predict] == 999]
    predict_df_not999 = predict_df[predict_df[predict] != 999]

    #The corresponding X and y obtained respectively
    y_train = predict_df_not999[predict]
    X_train = predict_df_not999.drop(predict, axis=1)
    X_predict = predict_df_999.drop(predict, axis=1)

    knn_clf = KNeighborsClassifier(n_neighbors=5)

    #The prediction result is obtained through KNN and updated to the original cell with Gene=999
    knn_clf.fit(X_train, y_train)
    all_df.loc[predict_df[predict] == 999, predict] = knn_clf.predict(X_predict)
