#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


titanic_df = pd.read_csv(r'C:\Users\seohyeon\.ipynb_checkpoints\titanic_train.csv.csv')
print('titanic 변수 type:' ,type(titanic_df))
titanic_df


# In[3]:


titanic_df.head(3)


# In[4]:


print('DataFrame 크기: ',titanic_df.shape)


# In[5]:


titanic_df.info()


# In[6]:


titanic_df.describe()


# In[7]:


value_counts = titanic_df['Pclass'].value_counts()
print(value_counts)


# In[8]:


titanic_pclass = titanic_df['Pclass']
print(type(titanic_pclass))


# In[9]:


titanic_pclass.head()


# In[10]:


value_counts = titanic_df['Pclass'].value_counts()
print(type(value_counts))
print(value_counts)


# In[11]:


import numpy as np


col_name1=['col1']
list1 = [1,2,3]
array1 = np.array(list1)
print('array1 shape:', array1.shape)
# 리스트를 이용해 DataFrame 생성.
df_list1 =pd.DataFrame(list1, columns=col_name1)
print('1차원 리스트로 만든 DataFrame:\n', df_list1)
#넘파이ndarray를 이용해 DataFrame 생성.
df_array1 = pd.DataFrame(array1, columns=col_name1)
print('1차원 ndarray로 만든 DataFrame:\n', df_array1)


# In[12]:


#3개의 칼럼명이 필요함.
col_name2=['col1', 'col2', 'col3']

#2행*3열 형태의 리스트와 ndarray 생성한 뒤 이를 DataFrame으로 변환.
list2 = [[1,2,3],
         [11,12,13]]
array2 = np.array(list2)
print('array2 shape:', array2.shape)
df_list2 = pd.DataFrame(list2, columns=col_name2)
print('2차원 리스트로 만든 DataFrame:\n', df_list2)
df_array2 = pd.DataFrame(array2, columns=col_name2)
print('2차원 ndarray로 만든 DataFrame:\n', df_array2)


# In[13]:


#key는 문자열 칼럼명으로 매핑, value는 리스트 형(또는 ndarray)칼럼 데이터로 매핑
dict = {'col1':[1, 11], 'col2':[2, 22], 'col3':[3,33]}
df_dict = pd.DataFrame(dict)
print('딕셔너리로 만든 DataFrame:\n', df_dict)


# In[14]:


#DataFrame을 ndarray로 변환
array3 = df_dict.values
print('df_dict.values 타입:', type(array3), 'df_dict.values shape:', array3.shape)
print(array3)


# In[15]:


#DataFrame을 리스트로 변환
list3= df_dict.values.tolist()
print('df_dict.values.tolist() 타입:',type(list3))
print(list3)

#DataFrame을 딕셔너리로 변환
dict3 = df_dict.to_dict('list')
print('\n df_dict.to_dict() 타입:', type(dict3))
print(dict3)


# In[16]:


titanic_df['Age_0']=0
titanic_df.head(3)


# In[17]:


titanic_df['Age_by_10'] = titanic_df['Age']*10
titanic_df['Family_No'] = titanic_df['SibSp'] + titanic_df['Parch']+1
titanic_df.head(3)


# In[18]:


titanic_df['Age_by_10'] = titanic_df['Age_by_10'] + 100
titanic_df.head(3)


# In[19]:


titanic_drop_df = titanic_df.drop('Age_0', axis=1)
titanic_drop_df.head(3)


# In[20]:


titanic_df.head(3)


# In[21]:


drop_result = titanic_df.drop(['Age_0', 'Age_by_10', 'Family_No'], axis=1, inplace=True)
print(' inplace=True 로 drop 후 반환된 값:', drop_result)
titanic_df.head(3)


# In[22]:


pd.set_option('display.width', 1000)
pd.set_option('display.max_colwidth', 15)
print('### before axis 0 drop ###')
print(titanic_df.head(3))
titanic_df.drop([0, 1, 2], axis=0, inplace=True)

print('### after axis 0 drop ###')
print(titanic_df.head(3))


# In[23]:


#원본 파일 다시 로딩
titanic_df = pd.read_csv(r'C:\Users\seohyeon\.ipynb_checkpoints\titanic_train.csv.csv')
#Index 객체 추출
indexes = titanic_df.index
print(indexes)
#Index 객체를 실제 값 array로 변환
print('Index 객체 array값:\n', indexes.values)


# In[24]:


print(type(indexes.values))
print(indexes.values.shape)
print(indexes[:5].values)
print(indexes.values[:5])
print(indexes[6])


# In[25]:


indexes[0]=5


# In[ ]:


series_fair = titanic_df['Fare']
print('Fare Series max 값:', series_fair.max())
print('Fair Series sum 값:', series_fair.sum())
print('sum() Fair Series:', sum(series_fair))
print('Fair Series + 3:\n', (series_fair + 3).head(3))


# In[ ]:


titanic_reset_df = titanic_df.reset_index(inplace=False)
titanic_reset_df.head(3)


# In[26]:


print('### before reset_index ###')
value_counts = titanic_df['Pclass'].value_counts()
print(value_counts)
print('value_counts 객체 변수 타입:', type(value_counts))
new_value_counts = value_counts.reset_index(inplace=False)
print('### After reset_index ###')
print(new_value_counts)
print('new_value_counts 객체 변수 타입:', type(new_value_counts))


# In[27]:


print('단일 칼럼 데이터 추출:\n', titanic_df['Pclass'].head(3))
print('\n여러 칼럼의 데이터 추출:\n', titanic_df[['Survived', 'Pclass']].head(3))
print('[]안에 숫자 index는 KeyError 오류 발생:\n', titanic_df[0])


# In[28]:


titanic_df[0:2]


# In[29]:


titanic_df[titanic_df['Pclass'] ==3].head(3)


# In[30]:


#ix[]연산자 지원 되지 않아 오류 발생
print('칼럼 위치 기반 인덱싱 데이터 추출:', titanic_df.ix[0,2])
print('칼럼 명 기반 인덱싱 데이터 추출:', titanic_df.ix[0, 'Pclass'])


# In[31]:


data = {'Name': ['Chulmin', 'Eunkyung', 'Jinwoong', 'Soobeom'],
        'Year': [2011, 2016, 2015, 2015],
        'Gender': ['Male', 'Female', 'Male', 'Male']
       }
data_df = pd.DataFrame(data, index=['one', 'two', 'three', 'four'])
data_df


# In[32]:


#data_df를 reset_index()로 새로운 숫자형 인덱스를 생성
data_df_reset = data_df.reset_index()
data_df_reset = data_df_reset.rename(columns={'index':'old_index'})

#인덱스값에 11을 더해서 1부터 시작하는 새로운 인덱스값 생성
data_df_reset.index = data_df_reset.index+1
data_df_reset


# In[33]:


data_df.iloc[0,0]


# In[34]:


#다음 코드는 오류를 발생시킵니다.
data_Df.iloc[0,'Name']


# In[35]:


#다음 코드는 오류를 발생시킵니다.
data_df.iloc['one', 0]


# In[36]:


data_df_reset.iloc[0,1]


# In[37]:


data_df.loc['one', 'Name']


# In[38]:


data_df_reset.loc[1, 'Name']


# In[39]:


#다음 코드는 오류를 발생합니다.
data_df_reset.loc[0, 'Name']


# In[40]:


print('위치 기반 iloc slicing\n', data_df.iloc[0:1, 0], '\n')
print('명칭 기반 loc slicing\n', data_df.loc['one':'two', 'Name'])


# In[41]:


print(data_df_reset.loc[1:2, 'Name'])


# In[42]:


titanic_boolean = titanic_df[titanic_df['Age']>60]
print(type(titanic_boolean))
titanic_boolean


# In[43]:


titanic_df[titanic_df['Age']>60][['Name', 'Age']].head(3)


# In[44]:


titanic_df.loc[titanic_df['Age']>60, ['Name', 'Age']].head(3)


# In[45]:


titanic_df[ (titanic_df['Age']>60) & (titanic_df['Pclass']==1) & 
            (titanic_df['Sex']=='female')]


# In[46]:


cond1 = titanic_df['Age']>60
cond2 = titanic_df['Pclass']==1
cond3 = titanic_df['Sex']=='female'
titanic_df[cond1&cond2&cond3]


# In[47]:


titanic_sorted = titanic_df.sort_values(by=['Name'])
titanic_sorted.head(3)


# In[48]:


titanic_sorted = titanic_df.sort_values(by=['Pclass', 'Name'], ascending=False)
titanic_sorted.head(3)


# In[49]:


titanic_df.count()


# In[50]:


titanic_df[['Age', 'Fare']].mean()


# In[51]:


titanic_groupby= titanic_df.groupby(by='Pclass')
print(type(titanic_groupby))


# In[52]:


titanic_groupby = titanic_df.groupby('Pclass').count()
titanic_groupby


# In[53]:


titanic_groupby = titanic_df.groupby('Pclass')[['PassengerId', 'Survived']].count()
titanic_groupby


# In[54]:


titanic_df.groupby('Pclass')['Age'].agg([max,min])


# In[55]:


agg_format={'Age':'max', 'SibSp':'sum', 'Fare':'mean'}
titanic_df.groupby('Pclass').agg(agg_format)


# In[56]:


titanic_df.isna().head(3)


# In[57]:


titanic_df.isna().sum()


# In[60]:


titanic_df['Cabin'] = titanic_df['Cabin'].fillna('C000')
titanic_df.head(3)


# In[61]:


titanic_df['Age'] = titanic_df['Age'].fillna(titanic_df['Age'].mean())
titanic_df['Embarked'] = titanic_df['Embarked'].fillna('S')
titanic_df.isna().sum()


# In[63]:


def get_square(a):
    return a**2


print('3의 제곱은:', get_square(3))


# In[64]:


lambda_square = lambda x : x**2
print('3의 제곱은:', lambda_square(3))


# In[65]:


a=[1, 2, 3]
squares = map(lambda x : x**2, a)
list(squares)


# In[66]:


titanic_df['Name_len']=titanic_df['Name'].apply(lambda x : len(x))
titanic_df[['Name', 'Name_len']].head(3)


# In[67]:


titanic_df['Child_Adult'] = titanic_df['Age'].apply(lambda x : 'Child' if x <=15 else 'Adult')
titanic_df[['Age', 'Child_Adult']].head(8)


# In[68]:


titanic_df['Age_cat'] = titanic_df['Age'].apply(lambda x : 'Child' if x<=15 else ('Adult' if x<=60
                                                                                     else 'Elderly'))
titanic_df['Age_cat'].value_counts()


# In[71]:


#나이에 따라 세분화된 분류를 수행하는 함수 생성.
def get_category(age):
    cat = ''
    if age <= 5: cat = 'Baby'
    elif age <= 12: cat = 'Child'
    elif age <= 18: cat = 'Teenager'
    elif age <= 25: cat = 'Student'
    elif age <= 35: cat = 'Young Adult'
    elif age <= 60: cat = 'Adult'
    else : cat = 'Elderly'
    
    
    return cat


# lambda 식에 위에서 생성한 get_category() 함수를 반환값으로 지칭.
# get_category(x)는 입력값으로 'Age' 칼럼 값을 받아서 해당하는 cat 반환
titanic_df['Age_cat'] = titanic_df['Age'].apply(lambda x: get_category(x))
titanic_df[['Age', 'Age_cat']].head()


# In[ ]:




