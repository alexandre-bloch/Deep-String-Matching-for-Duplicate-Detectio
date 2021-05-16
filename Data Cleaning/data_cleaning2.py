import pandas as pd
"""
For the Quora Question and restaurant datasets
"""
df = pd.read_csv("C:/Users/alexa/Desktop/Dad/ZZZZ/questions.csv")

####################################################################

df.replace(',','', regex=True, inplace=True)
df.replace('"','', regex=True, inplace=True)

df.question1.replace({r'[^\x00-\x7F]+':''}, regex=True, inplace=True)
df.question2.replace({r'[^\x00-\x7F]+':''}, regex=True, inplace=True)



####################################################################
grouped = df.groupby(df.is_duplicate)

match = grouped.get_group(1)
unmatch = grouped.get_group(0)

match.drop(['is_duplicate'],axis=1,inplace = True)
unmatch.drop(['is_duplicate'],axis=1,inplace = True)

match.drop(match.index[2500:],inplace = True)
unmatch.drop(unmatch.index[2500:],inplace = True)
####################################################################


match_s1 = match[['question1']].copy()
match_s2 = match[['question2']].copy()
unmatch_s1 = unmatch[['question1']].copy()
unmatch_s2 = unmatch[['question2']].copy()



match.to_csv("match.csv",index = False)
unmatch.to_csv("unmatch.csv",index = False)


match_s1.to_csv("match_s1.csv",index = False)
match_s2.to_csv("match_s2.csv",index = False)
unmatch_s1.to_csv("unmatch_s1.csv",index = False)
unmatch_s2.to_csv("unmatch_s2.csv",index = False)