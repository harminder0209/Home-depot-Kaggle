import pandas as pd
from sklearn.pipeline import Pipeline 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD


#Reading file train and test
df_train = pd.read_csv('C:\\Users\\Minal\\Desktop\\Harry\\dataset\\df_train_after_SC_Features_RE.csv', encoding="ISO-8859-1")
df_test = pd.read_csv('C:\\Users\\Minal\\Desktop\\Harry\\dataset\\df_test_after_SC_Features_RE.csv', encoding="ISO-8859-1")


num_train = df_train.shape[0]


#Concatinating test and train file do that tfidf and feature reduction can be done in 1 step
df_all = pd.concat((df_train, df_test), axis=0, ignore_index=True)


#removing the unwanted columns
df_all = df_all.drop(['Unnamed: 0','attr','product_info'],axis=1)



#tfidf and truncated svd initializer(to extract concept using Latent semantics analysis(LSA))
tfidf = TfidfVectorizer(ngram_range=(1, 2), stop_words='english')
svd = TruncatedSVD(n_components=100, random_state = 2016)  #n_components=100 to extract concepts using LSA

#creating pipeline to execute tfidf and svd in one step
pipe = Pipeline(steps=[('tfidf', tfidf), ('svd', svd)])


#Perform fit and trnsorm function of pipeline to convert text into vectors( number of features) and reducing them 
df_all["product_title"]=pipe.fit_transform(df_all["product_title"])
df_all["search_term"]=pipe.fit_transform(df_all["search_term"])
df_all["product_description"]=pipe.fit_transform(df_all["product_description"])
#df_all["brand"]=pipe.fit_transform(df_all["brand"])


#Separting train and test data 
df_train = df_all.iloc[:num_train]
df_test = df_all.iloc[num_train:]


#writing to csv
print('writing to csv')
df_train.to_csv("C:\\Users\\Minal\\Desktop\\Harry\\dataset\\df_train_after_tf-idf_brand.csv",index=False)
df_test.to_csv("C:\\Users\\Minal\\Desktop\\Harry\\dataset\\df_test_after_tf-idf_brand.csv",index=False)



