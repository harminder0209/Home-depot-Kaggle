import pandas as pd
from sklearn.ensemble import RandomForestRegressor, BaggingRegressor
#from sklearn.metrics.pairwise import cosine_similarity
#from sklearn.metrics import jaccard_similarity_score

df_train = pd.read_csv("C:\\Users\\Minal\\Desktop\\Harry\\dataset\\df_train_after_tf-idf.csv", encoding="ISO-8859-1")
df_test = pd.read_csv("C:\\Users\\Minal\\Desktop\\Harry\\dataset\\df_test_after_tf-idf.csv", encoding="ISO-8859-1")

#df_train.info()

#df_train["consine_similarity_search_and_title"]= jaccard_similarity_score(df_train["search_term"],df_train["product_title"])



df_train = df_train.drop(['brand'],axis=1)
df_test=df_test.drop(['brand'],axis=1)


rf = RandomForestRegressor(n_estimators=375, max_depth=10, random_state=20,n_jobs = -1)
clf = BaggingRegressor(rf, n_estimators=300, max_samples=0.1, random_state=25)

id_test = df_test['id']
y_train = df_train['relevance'].values
X_train = df_train.drop(['id','relevance'],axis=1).values
X_test = df_test.drop(['id','relevance'],axis=1).values

print("fitting starts")
clf.fit(X_train, y_train)
#
print("predicting starts")
y_pred = clf.predict(X_test)
###
pd.DataFrame({"id": id_test, "relevance": y_pred}).to_csv('C:\\Users\\Minal\\Desktop\\Harry\\output\\submission5_random_new parameter+grid search.csv',index=False)
##
