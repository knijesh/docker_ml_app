def score(score_data, model):
    target_col = 'ChurnValue'
    ignore_cols = ['ZipCode','SeniorCitizen','Gender','ChurnScore','CLTV','ChurnReason','ChurnLabel']
    id_cols = ['CustomerID']
    non_train_cols = id_cols + [target_col] + ignore_cols
    cats = list(score_data.select_dtypes(include=['object']).columns)
    cat_cols_for_transform = [x for x in cats if x not in non_train_cols]
    train_cols = [x for x in score_data.columns if x not in non_train_cols]
    
    print('-----Sample data-----')
    print(score_data[train_cols].head(2),'\n')

    print('-----Training columns-----')
    print(train_cols,'\n')

    print('-----Target column-----')
    print(target_col,'\n')

    print('-----Id columns-----')
    print(id_cols,'\n')
    
    print('-----Scoring data-----')
    score_data['p1'] = model.predict_proba(score_data[train_cols])[:,1]
    print('-----Scoring complete-----')
    return score_data
