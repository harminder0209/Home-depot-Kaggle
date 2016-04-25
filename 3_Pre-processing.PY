import numpy as np
import pandas as pd
from nltk.stem.snowball import SnowballStemmer
import re


stemmer = SnowballStemmer('english') # stemming method 


#reading files
print("reading files")
df_train = pd.read_csv('C:\\Users\\Minal\\Desktop\\Harry\\dataset\\train_spellcheck.csv', encoding="ISO-8859-1")
df_test = pd.read_csv('C:\\Users\\Minal\\Desktop\\Harry\\dataset\\test_spellcheck.csv', encoding="ISO-8859-1")
df_pro_desc = pd.read_csv('C:\\Users\\Minal\\Desktop\\Harry\\dataset\\product_descriptions.csv',encoding="ISO-8859-1")
df_attribute = pd.read_csv('C:\\Users\\Minal\\Desktop\\Harry\\dataset\\attributes.csv', encoding="ISO-8859-1")


#Creating a new feature brand

df_brand = df_attribute[df_attribute.name == "MFG Brand Name"][["product_uid", "value"]].rename(columns={"value": "brand"})


df_train["search_term"]= df_train["search_term"].astype(str)
num_train = df_train.shape[0]


# Function for removing regular expression
def string_edit(s): 
    if isinstance(s, str):
        s = re.sub(r"(\w)\.([A-Z])", r"\1 \2", s) #Split words with a.A
        s = s.lower()
        s = s.replace("  "," ")
        s = s.replace(",","") #could be number / segment later
        s = s.replace("$"," ")
        s = s.replace("?"," ")
        s = s.replace("-"," ")
        s = s.replace("//","/")
        s = s.replace("..",".")
        s = s.replace(" / "," ")
        s = s.replace(" \\ "," ")
        s = s.replace("."," . ")
        s = re.sub(r"(^\.|/)", r"", s)
        s = re.sub(r"(\.|/)$", r"", s)
        s = re.sub(r"([0-9])([a-z])", r"\1 \2", s)
        s = re.sub(r"([a-z])([0-9])", r"\1 \2", s)
        s = s.replace(" x "," xbi ")
        s = re.sub(r"([a-z])( *)\.( *)([a-z])", r"\1 \4", s)
        s = re.sub(r"([a-z])( *)/( *)([a-z])", r"\1 \4", s)
        s = s.replace("*"," xbi ")
        s = s.replace(" by "," xbi ")
        s = re.sub(r"([0-9])( *)\.( *)([0-9])", r"\1.\4", s)
        s = re.sub(r"([0-9]+)( *)(inches|inch|in|')\.?", r"\1in. ", s)
        s = re.sub(r"([0-9]+)( *)(foot|feet|ft|'')\.?", r"\1ft. ", s)
        s = re.sub(r"([0-9]+)( *)(pounds|pound|lbs|lb)\.?", r"\1lb. ", s)
        s = re.sub(r"([0-9]+)( *)(square|sq) ?\.?(feet|foot|ft)\.?", r"\1sq.ft. ", s)
        s = re.sub(r"([0-9]+)( *)(cubic|cu) ?\.?(feet|foot|ft)\.?", r"\1cu.ft. ", s)
        s = re.sub(r"([0-9]+)( *)(gallons|gallon|gal)\.?", r"\1gal. ", s)
        s = re.sub(r"([0-9]+)( *)(ounces|ounce|oz)\.?", r"\1oz. ", s)
        s = re.sub(r"([0-9]+)( *)(centimeters|cm)\.?", r"\1cm. ", s)
        s = re.sub(r"([0-9]+)( *)(milimeters|mm)\.?", r"\1mm. ", s)
        s = s.replace("°"," degrees ")
        s = re.sub(r"([0-9]+)( *)(degrees|degree)\.?", r"\1deg. ", s)
        s = s.replace(" v "," volts ")
        s = re.sub(r"([0-9]+)( *)(volts|volt)\.?", r"\1volt. ", s)
        s = re.sub(r"([0-9]+)( *)(watts|watt)\.?", r"\1watt. ", s)
        s = re.sub(r"([0-9]+)( *)(amperes|ampere|amps|amp)\.?", r"\1amp. ", s)
        s = s.replace("  "," ")
        s = s.replace(" . "," ")
        return s
    else:
        return "null"



#stemming function
def str_stemmer(s):
	return " ".join([stemmer.stem(word) for word in s.lower().split()])



#common word function
def str_common_word(str1, str2):
	return sum(int(str2.find(word)>=0) for word in str1.split())




#concatinating train and test set so that pre-porcessing can be done on them in one step
df_all = pd.concat((df_train, df_test), axis=0, ignore_index=True) 


#merging train, test and product description on product_id by left join
print('merging')
df_all = pd.merge(df_all, df_pro_desc, how='left', on='product_uid')
df_all = pd.merge(df_all, df_brand, how = 'left', on = 'product_uid')


df_all['brand']=df_all['brand'].astype(str)


# Removing regular expression from search item , product title, and product description
print("regurlar expression")
df_all['search_term'] = df_all['search_term'].map(lambda x:string_edit(x))
df_all['product_title'] = df_all['product_title'].map(lambda x:string_edit(x))
df_all['product_description'] = df_all['product_description'].map(lambda x:string_edit(x))


#performing stemming function on search term, product title, and product description
print("stemming")
df_all['search_term'] = df_all['search_term'].map(lambda x:str_stemmer(x))
df_all['product_title'] = df_all['product_title'].map(lambda x:str_stemmer(x))
df_all['product_description'] = df_all['product_description'].map(lambda x:str_stemmer(x))


##creating a new feature in length of query, lenght of brand, word in title, word in description, brand_in_search, 
#  ratio_brand,ratio_description,last_word_title_match,last_word_description_match, first_word_title_match, first_word_description_match 
print('creating features')
df_all['len_of_query'] = df_all['search_term'].map(lambda x:len(x.split())).astype(np.int64)
df_all['length_of_brand'] = df_all['brand'].map(lambda x:len(x.split())).astype(np.int64)
df_all['product_info'] = df_all['search_term']+"\t"+df_all['product_title']+"\t"+df_all['product_description']
df_all['word_in_title'] = df_all['product_info'].map(lambda x:str_common_word(x.split('\t')[0],x.split('\t')[1]))
df_all['word_in_description'] = df_all['product_info'].map(lambda x:str_common_word(x.split('\t')[0],x.split('\t')[2]))
df_all['attr'] = df_all['search_term']+"\t"+df_all['brand']
df_all['brand_in_search'] = df_all['attr'].map(lambda x:str_common_word(x.split('\t')[0],x.split('\t')[1]))
df_all['ratio_brand'] = df_all['brand_in_search']/df_all['length_of_brand']
df_all['ratio_title'] = df_all['word_in_title']/df_all['len_of_query']
df_all['ratio_description'] = df_all['word_in_description']/df_all['len_of_query']
df_all['last_word_title_match'] = df_all['product_info'].map(lambda x:str_common_word(x.split('\t')[0].split(" ")[-1],x.split('\t')[1]))
df_all['last_word_description_match'] = df_all['product_info'].map(lambda x:str_common_word(x.split('\t')[0].split(" ")[-1],x.split('\t')[2]))
df_all['first_word_title_match'] = df_all['product_info'].map(lambda x:str_common_word(x.split('\t')[0].split(" ")[0],x.split('\t')[1]))
df_all['first_word_description_match'] = df_all['product_info'].map(lambda x:str_common_word(x.split('\t')[0].split(" ")[0],x.split('\t')[2]))



#spliting data into test and train again
print('splitting data')
df_train = df_all.iloc[:num_train]
df_test = df_all.iloc[num_train:]


#writing pre-processed file
print('writing to csv')
df_train.to_csv("C:\\Users\\Minal\\Desktop\\Harry\\dataset\\df_train_after_SC_Features_RE.csv",index=False)
df_test.to_csv("C:\\Users\\Minal\\Desktop\\Harry\\dataset\\df_test_after_SC_Features_RE.csv",index=False)

