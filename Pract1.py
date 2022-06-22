# Pandas Practice
import pandas as pd
import numpy as np

'''Pandas is used to analyze the data. It is used for working with data sets. Having functions to analyzing, cleaning, exploring and manipulating data.
It is created by Wes McKinney in 2008.
Pandas are also able to delete the rows that are not relevant, or contains wrong values, like empty or Null values. This is called cleaning the data.
To install Pandas  pip install pandas
Importing pandas  import pandas as pd
'''

'''Series.index – Defines the index of the Series.
Series.shape – It returns a tuple of shape of data.
Series.dtype – It returns the data type of the data.
Series.size – It returns the size of the data.
Series.empty – It returns True if Series object is empty, otherwise returns false.
Series.hasnans – It returns True if there are any NaN values, otherwise returns false.
Series.nbytes – It returns the number of bytes in the data.
Series.ndim – It returns the number of dimensions in the data.
Series.itemsize – It returns the size of the  datatype of item.
'''
'''import mysql.connector
conn = mysql.connector.connect(user='root', password='ViAli@161SK#', host='localhost', database='jaquar',
                               auth_plugin='mysql_native_password')
print(conn.is_connected())'''

# Create DataFrame
'''mydataset = {'cars':['BMW','Lumbargini','Volvo'],
             'passings':[3,7,2]}
myvar =pd.DataFrame(mydataset)
print(myvar)'''

# To check the pandas version
#print(pd.__version__)

# Create Series
'''a=[1,2,3,4]
se = pd.Series(a)
print(se)'''

''' 
info = np.array(['P','a','n','d','a','s'])
a = pd.Series(info)
print(a)'''

#d = pd.Series()
#print(d)

#Series using dictonary
'''info = {'x':10,'y':20,'z':60}
df = pd.Series(info)
print(df)'''

'''a = pd.Series(4,index=[0,1,2,3])
print(a)'''


'''x = pd.Series(data=[2,4,6,8])
y = pd.Series(data=[11.2,18.6,22.5],index=['a','b','c'])

print(x.index)
print(x.values)
print(y.index)
print(y.values)
print(x.dtype)
print(x.itemsize)
print(y.dtype)
print(y.itemsize)
print(x.shape)
print(y.shape)'''

'''d1 = pd.Series(data=[1,2,3,np.NaN])
d = pd.Series(data=[1,2,3,4],index=['a','b','c','d'])
d2 = pd.Series()
print(d1.hasnans,d2.hasnans,d.hasnans)
print(d1.empty,d2.empty,d.empty)
print(len(d1),len(d2),len(d))
print(d1.count(),d2.count(),d.count())'''

# map() function
'''a = pd.Series(['Java','C','C++',np.nan])
s = a.map({'Java':'Core'})
print(s)'''


'''a = pd.Series(['Java','C','C++',np.nan])
a.map({'Java':'Core'})
s = a.map("I like {} ".format,na_action='ignore')
print(s)'''

'''a = pd.Series(['Java','C','C++','Python'])
corre = {"Java":"alien",'C':'not alien','C++':'fog','Python':'rod'}

s = a.map(corre)
print(s)'''

# std() standard deviation
'''a = np.std([1,4,3,2,7])
print(a)'''

'''info = {'Name':["Parker","Smith","John",'William'],
        'sub1_marks':[45,60,34,65],
        'sub2_marks':[90,87,45,67]}
a = pd.DataFrame(info)
s = a.std()
print(s)'''

# to_frame()    this function is used to convert series object to Dataframe
'''s = pd.Series(['a','b','c'],name='vals')
a = s.to_frame()
print(a)
print(a.dtypes)'''

'''emp = ['Parker','John','Smith','William']
id = [102,103,104,105]
emp_series = pd.Series(emp)
id_series = pd.Series(id)
frame = {'Employee':emp_series,'ID':id_series}
result = pd.DataFrame(frame)
print(result)'''

# unique() to find the unique elements present in columns.
'''s = {'id':[1,2,3,4,5],'Name':['Sam','Ram','Bheem','Raheem','John']}

a = s['id'].unique()
print(a)'''

# value_counts() it returns a series that contains  count of unique values
'''index = pd.Index([2,1,1,np.nan,3])
s = index.value_counts()
print(s)'''

'''index = pd.Index([2,1,1,np.nan,3])
s = index.value_counts(dropna=False)
print(s)'''

# labels:- Labels are indexes
'''a = [1,2,3,4]
myvar = pd.Series(a,index=['w','x','y','z'])
print(myvar)'''

'''a = {'day1':40,'day2':50,'day3':30}
myvar = pd.Series(a)
print(myvar)'''

'''a = {'day1':40,'day2':50,'day3':30}
myvar = pd.Series(a,index=['day1','day2'])
print(myvar)'''

# DataFrame
'''info = {'one':pd.Series([1,2,3,4,5],index=['a','b','c','d','e']),
        'two':pd.Series([1,2,3,4,5,6],index=['a','b','c','d','e','f'])}
info = pd.DataFrame(info)
info['three'] = pd.Series([1,2,3,4],index=['a','b','c','d'])
info['four'] = info['two']+info['three']
print(info)'''

'''df = pd.DataFrame()
print(df)'''

'''data = {"calories":[40,380,490],'Duration':[34,45,56]}
myvar = pd.DataFrame(data)
print(myvar)

info = {'one':pd.Series([1,2,3,4,5,6],index=['a','b','c','d','e','f']),
        'two':pd.Series([1,2,3,4,5,6,7,8],index=['a','b','c','d','e','f','g','h'])}

df = pd.DataFrame(info)
print(df)'''

'''info = {'one':pd.Series([1,2,3,4,5,6],index=['a','b','c','d','e','f']),
        'two':pd.Series([1,2,3,4,5,6,7,8],index=['a','b','c','d','e','f','g','h'])}
d1 = pd.DataFrame(info)
print(d1['one'])'''

'''info = {'one':pd.Series([1,2,3,4,5,6],index=['a','b','c','d','e','f']),
        'two':pd.Series([1,2,3,4,5,6,7,8],index=['a','b','c','d','e','f','g','h'])}
d1 = pd.DataFrame(info)
d1['three'] = pd.Series([1,2,3,4],index=['a','b','c','d'])
print(d1)
del d1['one']
print(d1)
d1.pop('two')
print(d1)'''

# Row selection : we can select any row by passing row lable to a "loc" function
'''info = {'one':pd.Series([1,2,3,4,5,6],index=['a','b','c','d','e','f']),
        'two':pd.Series([1,2,3,4,5,6,7,8],index=['a','b','c','d','e','f','g','h'])}
df = pd.DataFrame(info)
print(df.loc['c'])'''

# Row selection by integer location 'iloc'
'''info = {'one':pd.Series([1,2,3,4,5,6],index=['a','b','c','d','e','f']),
        'two':pd.Series([1,2,3,4,5,6,7,8],index=['a','b','c','d','e','f','g','h'])}
df = pd.DataFrame(info)
print(df.iloc[3])'''

'''info = {'one':pd.Series([1,2,3,4,5,6],index=['a','b','c','d','e','f']),
        'two':pd.Series([1,2,3,4,5,6,7,8],index=['a','b','c','d','e','f','g','h'])}
df = pd.DataFrame(info)
print(df.iloc[2:5])'''

#Addition of rows using append function
'''d = pd.DataFrame([[7,8],[9,10]],columns=['x','y'])
d2 = pd.DataFrame([[11,12],[13,14]],columns=['x','y'])
d = d.append(d2)
print(d)'''

# Deleting rows
'''d = pd.DataFrame([[7,8],[9,10]],columns=['x','y'])
d2 = pd.DataFrame([[11,12],[13,14]],columns=['x','y'])
d = d.append(d2)
df = d.drop(0)
print(df)'''

# append() function to add rows of the dataframe to end of the given dataframe.
'''info1 = pd.DataFrame({"x":[25,15,12,19],"y":[47,24,17,29]})
info2 = pd.DataFrame({"x":[25,15,12],"y":[47,24,17],"z":[38,12,45]})
inf = info1.append(info2,ignore_index= True)
print(inf)'''

#apply() it allows the user to pass a function and apply it to every single value of the Pandas Series.

'''data = pd.DataFrame({"power_level": [12000, 16000, 4000, 1500, 3000, 2000, 1600, 2000, 300],
                     'uniform color': ["orange", "blue", "black", "orange", "purple", "green", "orange", "orange",
                                       "orange"],
                     'species': ["saiyan", "saiyan", "saiyan", "half saiyan", "namak", "human", "human", "human",
                                 "human"]},
                    index=["Goku", "Vegeta", "Nappa", "Boppa", "Piccolo", "Tien", "Yamcha", 'hell', 'Hrithik']
                    )

#print(data)


def my_func(x,h,l):
    if x >h:
        return ('high')
    if x> l:
        return ('med')
    return ('low')


s = data["power_level"].apply(my_func,args=[10000,2000])
print(s)'''

# aggregate() function to perform specific operation on particular field

'''data = {'x':[50,40,30],'y':[300,1112,42]}
df = pd.DataFrame(data)
x = df.aggregate(["sum"])
print(x)'''

# assign() method adds a new column to an existing DataFrame, this method doesn't change the original dataframe.
'''data = {"age":[16,14,10],
    "qualified":[True,True,True]
}
df = pd.DataFrame(data)
new_df = df.assign(name=['Emmly','Sushanth','Singh'])
print(new_df)'''

# astype() this method is used to change the datatype of elements
'''data = {"Duration":[50,40,45],
    "Pulse":[109,117,342],
    "Calories":[409.1,479.5,983.2]
}

df = pd.DataFrame(data)
new_df = df.astype('int64')
print(new_df)'''

# counts() it is used to count the number of not empty values.
'''data = {"Duration":[50,40,None,None,90,20],
    "Pulse":[109,140,110,125,138,170]
}
df = pd.DataFrame(data)
print(df.count())'''

# cut() this method is used to convert a continous variable to a catogerical variable, to define the data in the range

'''dict = {'Customer_name':['Gopi','Ram','Chandu','Mani','Ahammad','Anil','Nisar'],
    'City':['Bhubaneshwar','Kolkata','Guntur','Hyderabad','Bangalore','Chennai','Nellore'],
    'Age':[18,25,30,40,50,80,70],
    'Amount_Purchased':[100,200,300,400,500,600,700]
}
df = pd.DataFrame(dict)
#print(df)
# Defining bins as 0 to 18, 18 to 25, 25 to 35, 35 to 55, 55 to 75, 75 to 100
bins=[0,18,25,35,55,75,100]
group_names = ['Teen','Young','Adult','Mid Adult','Old','Senior Citizen']
df['Age Group'] = pd.cut(df.Age,bins,labels=group_names)
print(df)'''


# describe() this method returns description of the data in the DataFrame.
'''data = [[10,18,11],[13,15,8],[9,20,3]]
df = pd.DataFrame(data)
print(df.describe())'''

# drop_duplicates() this method is used to drop duplicates
'''data = {"name":["Sally","Wally","Marry","Marry"],
    "age":[50,40,30,30],
    "qualified":[True,False,False,False]
}
df = pd.DataFrame(data)
ddf = df.drop_duplicates()
print(ddf)'''

# groupby() - This method allows you to group your data.
'''data = {'co2':[95,90,99,104,105,94,99,104],
'model':['Citigo','Fabia','Fiesta', 'Rapid', 'Focus', 'Mondeo', 'Octavia', 'B-Max'],
'car':['Skoda', 'Skoda', 'Ford', 'Skoda', 'Ford', 'Ford', 'Skoda', 'Ford']
}
df = pd.DataFrame(data)
print(df.groupby(["car"]).mean())'''

# head() method is used to return specified number of rows, by default it returns 5

'''data = pd.read_csv('data.csv')
print(data.head())
print("------------------------")
print(data.head(10))'''

# iterrows() : This method generates an iterator object of the dataframe.
'''data = {"firstname":["Sally","Mary","John"],
    "age":[50,40,30]
}

df = pd.DataFrame(data)
for index,row in df.iterrows():
    print(row['firstname'],index)'''

# join() This method inserts columns from another dataframe , or series.
'''data1 = {"name":["Sall","Mary","John"],
    "age":[50,40,30]
}
data2 = {
    "qualified":[True,False,False]
}

df1 = pd.DataFrame(data1)
df2 = pd.DataFrame(data2)
df = df1.join(df2)
print(df)'''

# mean() method return a series with the mean value of each column.
'''data = [[1,1,2],[6,4,2],[4,2,1],[4,2,3]]
df = pd.DataFrame(data)
print(df.mean())'''

# melt() this method reshapes the DataFrame into a long table with one row for each column
'''data = pd.read_csv('data.csv')
newdf = data.melt()
print(newdf.to_string())'''

# merge() this method updates the content of two dataframes by merging them together, using specified methods.
'''data1 = {"name":["Sally","Mary","John"],
"age":[50,40,30]}
data2 = {"name":["Sally","Peter","Micky"],
    "age":[77,44,22]
}

df1 = pd.DataFrame(data1)
df2 = pd.DataFrame(data2)
newdf = df1.merge(df2,how='right')
print(newdf)'''

# pivot_table() The pandas pivot_table  is used to caluclate aggregate nd summarize your data.
'''info = pd.DataFrame({'P':['Smith','Jhon','William','Rinku'],
    'Q':['Python','C','C++','Java'],
    'R':[19,24,22,25]
})

table = pd.pivot_table(info,index=['P','Q'])
print(table)'''

# query() the query method allows you to query the DataFrame , the query method takes a query expression as a string parameter.
'''data = {'name':["Sally","Mary","John"],
    "age":[5,40,30]
}

df = pd.DataFrame(data)
print(df.query('age>35'))'''


# rename() this method allows you to change the row indexes and column labels. this doesn't make changes to the original dataframe.

'''data = {"age":[50,40,30],
    "qualified":[True,False, False],
}
idx = ["Sally","Mary","John"]
df = pd.DataFrame(data,index=idx)
#print(df)
new_df = df.rename({"Sally":"Pete","Mary":"Patrik","John":"Paula"})
print(new_df)'''

# Sample() this method returns a specfied number of random rows.
'''df = pd.read_csv('data.csv')
print(df.sample())'''

# shift() if we want to shift your column or substract the column value with the previous row value from the DataFrame.
'''info = pd.DataFrame({'a_data':[45,28,39,32,18],
    'b_data':[26,37,41,35,45],
    'c_data':[22,19,11,25,16]
})
p = info.shift(periods=2)
print(p)'''

# sort_index() This method sorts the DataFrame by the index.
'''data = {"age":[50,40,30,40,20,10,30],
    "qualified":[True,False,False,False,False,True,True]
}
idx = ["Mary","Sall","Emily","Tobins","Linus","John","Peter"]

df = pd.DataFrame(data,index=idx)
newdf = df.sort_index()
print(newdf)'''

'''data = {"age":[50,40,30,40,20,10,30],
    "qualified":[True,False,False,False,False,True,True]
        }
df = pd.DataFrame(data)
new_df = df.sort_values(by='age')
print(new_df)'''

# sum() this method adds all values in each column and returns the sum for each column.
'''data =[[10,18,11],[13,15,8],[9,20,3]]
df = pd.DataFrame(data)
print(df.sum())'''

# to_excel() we can export data from dataframe to excel file
'''info_marks = pd.DataFrame({'name':['Parker','Smith','William','Terry'],
    'Maths':[78,84,67,72],
    'Science':[89,92,61,77],
    'English':[72,75,64,82]
})

writer = pd.ExcelWriter('output.xlsx')
info_marks.to_excel(writer)
writer.save()'''

# transform() this method allows you to execute a function for each value of the dataframe.

'''def euc(x):
    return x*10
data = {'for1':[2,6,3],
    'for5':[8,20,12]
}

df = pd.DataFrame(data)
newdf = df.transform(euc)
print(newdf)'''

# transpose() this method transform the columns in to rows and rows into columns.
'''data = {'age':[50,40,30,40,20,10,30],
    'qualified':[True,False,False,False,False,True,True]
}

df = pd.DataFrame(data)
newdf = df.transpose()
print(newdf)'''

# where() this method replaces the values of the rows where the condition evaluates to false.
'''data = {'age':[50,40,30,40,20,10,30],
        'qualified':[True,False,False,False,False,True,True]}

df = pd.DataFrame(data)
newdf = df.where(df["age"]>30)
print(newdf)'''

# to add a column to the Pandas csv file
'''data = pd.read_csv('data.csv')
data['Age'] = 24
print(data.head())'''

# insert() add a new column at any position in an existing DataFrame using a method name.
'''data = pd.read_csv('data.csv')
data.insert(2,column='Department',value='B.SC')
print(data.head())'''

# Convert Pandas dataframe to numpy array
'''info = pd.DataFrame([[17,62,35],[25,36,54],[42,20,15],[48,62,76]],
columns=['x','y','z'])

print("dataframe\n --------\n",info)
arr = info.to_numpy()
print(arr)'''

# change data from datafrmae to read_csv
'''data = {'Name':['Smith','Parker'],'ID':[101,pd.NaT],'Language':[pd.NaT,'JavaScript']}
info = pd.DataFrame(data)
print('DataFrame values:\n',info)
print()
csv_data = info.to_csv(na_rep= 'None')
print(csv_data)
print()'''

# read csv
'''df = pd.read_csv('data.csv')
print(df)'''

# read json
'''df = pd.read_json('data1.json')
print(df)'''

# Pandas concatenation
'''a_data = pd.Series(['p','q'])
b_data = pd.Series(['r','s'])
d = pd.concat([a_data,b_data])
print(d)'''


'''one = pd.DataFrame({
    'Name':['Parker','Smith','Allen','John','Parker'],
    'subject_id':['sub1','sub2','sub4','sub5','sub6'],
    'Marks_scored':[98,90,87,69,78]
},index=[1,2,3,4,5])
two = pd.DataFrame({'Name':['Billy','Brain','Bran','Bryce','Betty'],
    'subject_id':['sub2','sub4','sub3','sub6','sub5'],
    'Marks_scored':[89,80,79,97,88]
},index=[1,2,3,4,5])

print(one.append(two))'''

# data processing
'''info = pd.Series([11,14,17,24,19,32,34,27],index = [['x','x','x','x','y','y','y','y'],['obj1','obj2','obj3','obj4','obj1','obj2','obj3','obj4']])
print(info)'''

# unstack() the data
'''info = pd.Series([11,14,17,24,19,32,34,27],index=[['x','x','x','x','y','y','y','y'],['obj1','obj2','obj3','obj4','obj1','obj2','obj3','obj4']])
print(info.unstack(0))'''

# stack() is used to convert the column index to row index.
'''info = pd.Series([11,14,17,24,19,32,34,27],index=[['x','x','x','x','y','y','y','y'],['obj1','obj2','obj3','obj4','obj1','obj2','obj3','obj4']])
o = info.unstack(0)
print(o.stack())'''

#corr() : This is used to find the correlation of each column in a DataFrame.
'''data = {'Duration':[50,40,45],
    'Pulse':[109,117,110],
    'Calories':[409.1,479.5,340.8]
}
df = pd.DataFrame(data)
print(df.corr())'''

# replace() this method replaces the specified value with another specified value
'''data = {"name":["Bill","Bob","Betty"],
"age":[50,50,30],
"qualified":[True,False,False]
}
df = pd.DataFrame(data)
newdf = df.replace(50,60)
print(newdf)'''

# iloc() this property gets,or sets the value(s), of the specified indexes. specify both rows and column with an index.
'''data = [[50,True],[40,False],[30,False]]
df = pd.DataFrame(data)
print(df)
print(df.iloc[1,0])'''

#isin() this method checks if the DataFrame contains the specified values.
'''data = {'name':["Sally","Mary","John"],
    "age":[50,40,30]
}
df = pd.DataFrame(data)
print(df.isin([50,40]))'''

# loc() this property gets or sets the values of the specified labels. Specify both row and column with a label to access more than one row.
'''data = [[50,True],[40,False],[30,False]]
label_rows = ["Sally","Mary","John"]
label_cols = ["age","qualified"]
df = pd.DataFrame(data,label_rows,label_cols)
print(df.loc["Mary","age"])
print(df.loc["Sally":"John"])'''

# dropna() it is used to drop the null values from the dataframe.
'''data = {'Name':["Sam",'Ram','Bheem','Satyam','Shivam','Sundaram','Raheem'],
        'Age':[25,46,56,67,66,55,np.nan],
        'Branch':['ECE','CSE','ME','EEE','IT',np.nan,'CIVIL']}

df = pd.DataFrame(data)
print(df.dropna())
print(df.dropna(axis=1))'''

# fillna() it replaces all null values with x.
'''data = {'Name':["Sam",'Ram','Bheem','Satyam','Shivam','Sundaram','Raheem'],
        'Age':[25,46,56,67,66,55,np.nan],
        'Branch':['ECE','CSE','ME','EEE','IT',np.nan,'CIVIL']}
df = pd.DataFrame(data)
print(df.fillna(df.mean()))
print(df.fillna(67))'''

# replace(1,'one) it replaces all the values equal to 1 with 'one'
'''data = {'Name':["Sam",'Ram','Bheem','Satyam','Shivam','Sundaram','Raheem'],
        'Age':[25,46,56,67,66,55,np.nan],
        'Branch':['ECE','CSE','ME','EEE','IT',np.nan,'CIVIL']}
df = pd.DataFrame(data)
f = df.replace(56,'Fifty Six')
print(f)'''

# rename() is used to rename the column names
'''data = {'Name':["Sam",'Ram','Bheem','Satyam','Shivam','Sundaram','Raheem'],
        'Age':[25,46,56,67,66,55,np.nan],
        'Branch':['ECE','CSE','ME','EEE','IT',np.nan,'CIVIL']}
df = pd.DataFrame(data)
s = df.rename(columns={'Branch':'Dept'})
print(s)'''

# set_index() is used to change the index name by selecting the specific column
'''data = {'Name':["Sam",'Ram','Bheem','Satyam','Shivam','Sundaram','Raheem'],
        'Age':[25,46,56,67,66,55,np.nan],
        'Branch':['ECE','CSE','ME','EEE','IT',np.nan,'CIVIL']}
df = pd.DataFrame(data)
d = df.set_index('Branch')
print(d)'''

"""Python Cheat Sheet:- 
df : Refer to any Pandas DataFrame object.
s : Refers to any Pandas Series object. 
Importing Data :
pd.read_csv(filename): It read the data from CSV file.
Pd.read_table(filename): It is used to read the data from delimiter text file.
Pd.read_excel(filename): It reads the data from an excel file.
Pd.read_sql(query,connection_object): It read the data from a SQL table/database.
Pd.read_json(json_string): It read the data from a JSON formatted string URL or file.
Pd.read_html(url): It parses an html URL, string or the  file and extract the tables to a list of dataframes.
Pd.read_clipboard(): It takes the contents of clipboard and passes it to the read_table() function.
Pd.DataFrame(dict): From the dict, keys for the columns names values for the data as lists.
Exporting Data: 
df.to_csv(filename): It writes to a CSV file.
df.to_excel(filename): It writes to an excel file.
df.to_sql(table_name,connection_object): It writes to a SQL table.
df.to_json(filename): It write to a file in JSON format.
Create Test Objects:
pd.DataFrame(np.random.rand(7,18)): Refers to 18 columns and 7 rows of random floats.
pd.Series(my_list): It create a Series from an iterable my_list.
df.index = pd.date_range(‘1940/1/20’,periods = df.shape[0])
Viewing/Inspecting Data:
df.head(n): It returns first n rows of the DataFrame.
Df.tail(n): It returns last n rows of the DataFrame.
df.shape(): It returns number of rows and columns.
Df.info() : it returns index, Datatype, and memory information.
s.value_counts(dropna=False): It views unique values and counts.
Df.apply(pd.Series.value_counts): It refers to the unique values and counts for all the columns.
Selection:
Df[col1]: It returns column with the label col as Series.
Df[[col1,col2]]: It returns columns as a new DataFrame.
s.iloc[0]: It select by the position.
s.loc[‘index_one’]:It select by the index.
Df.iloc[0,:]: It returns first row.
Df.iloc[0,0]: It returns the first element of first column.
DataCleaning: 
df.columns = [‘a’,’b’,’c’]: It rename the columns.
Pd.isnull(): It checks for the null values and return the Boolean array.
Pd.notnull(): It is opposite of pd.isnull()
Df.dropna() : It drop all the rows that contain the null values.
Df.dropna(axis=1): It drops all the columns that contains null values.
Df.dropna(axis=1,thresh=n): It drops all the rows that have less than n non null values.
Df.fillna(x): It replaces all null values with x.
s.fillna(s.mean()): It replaces all the null values with the mean (the mean can be replaced with almost any function from the statistics module).
s.astype(float): It converts the data type of series to float.
s.replace(1,’one’): It replaces all the values equal to 1 with ‘one’.
s.replace([1,3][‘one’,’three’]): It replaces all 1 with ‘one’ and 3 with ‘three’.
Df.rename(columns=lambda x:x+1): it renames mass of the columns.
Df.rename(columns={‘old_name’:’new_name’}): It consist selective renaming.
Df.set_index(‘column_one’): Used for changing the index.
Df.rename(index=lambda x:x+1): It renames mass of the index.
Filter, sort & groupby: 
Df[df[col]>0.5]: Return the rows where column col is greater than 0.5.
Df[(df[col]>0.5) & (df[col]<0.7)]: Returns the rows where 0.7>col>0.5
Df.sort_values(col1): It sorts the values by col1 in ascending order.
Df.sort_values(col2,ascending=False): It sorts the values by col2 in descending order.
Df.sort_values([col1,col2],ascending=[True,False]): It sorts the values by col1 in ascending order and col2 in descending order.
Df.groupby(col1): Returns a groupby object for the values from one column.
Df.groupby([col1,col2]): Returns a group by object for values from multiple columns.
Df.pivot_table(index=col1,values=[col2,col3],aggfunc=mean): It creates the pivot table that groups by col1 and calculate mean of col2 and col3.
Df.groupby(col1).agg(np.mean): It calculate the average across all the columns for every unique col1 group.
Df.apply(np.mean) : Its task is to apply the function np.mean() across each column.
Df.apply(np.max,axis=1): Its task is to apply the function np.max() across each row.
Join/Combine:
Df1.append(df2): Its task is to add the rows in df1 to the end of df2 (columns should be identical).
Pd.concat([df1,df2],axis=1): Its task is to add the columns in df1 to the end of df2(rows should be identical).
Df1.join(df2,on=col1,how=’inner’): SQL- style join the columns in df1 with the columns on df2 where the rows for col have identical values, ‘how’ can be of ‘left’ , ‘right’, ‘outer’, ‘inner’.
Statistics:
Df.describe() : It returns the summary statistics for the numerical columns.
Df.mean() : It returns the mean of all columns.
Df.corr(): It returns the correlation between the columns in the DataFrame.
Df.count() : It returns the count of all the non-null values in each dataframe column.
Df.max(): It returns the highest value from each of the columns.
Df.min() : It returns the lowest value from each of the columns.
Df.median() : It returns the median from each of the columns.
Df.std(): It returns the standard deviation from each of the column.
"""

# sort_values() is used to sort the columns in acsending or descending
'''data = {'Name':["Sam",'Ram','Bheem','Satyam','Shivam','Sundaram','Raheem'],
        'Age':[25,46,56,67,66,55,np.nan],
        'Branch':['ECE','CSE','ME','EEE','IT',np.nan,'CIVIL']}
df = pd.DataFrame(data)
d = df.sort_values(['Age','Branch'],ascending=[True,False])
print(d)'''


# groupby() is used to return a groupby object for the values from one column.
'''data = {'Name':["Sam",'Ram','Bheem','Satyam','Shivam','Sundaram','Raheem'],
        'Age':[25,46,56,67,66,55,np.nan],
        'Branch':['ECE','CSE','ME','EEE','IT',np.nan,'CIVIL']}
df = pd.DataFrame(data)
d = df.groupby('Age')
print(d)'''

# reindex() is used to change the index values.
'''info = pd.DataFrame({"P":[4,7,1,8,9],
    "Q":[6,8,10,15,11],
    "R":[17,13,12,16,18],
    "S":[15,19,7,21,9]
},
index = ["Parker","William","Smith","Terry","Phill"])
info = info.reindex(["A","B","C","D","E"])
print(info)'''


'''info = pd.DataFrame({"P":[4,7,1,8,9],
    "Q":[6,8,10,15,11],
    "R":[17,13,12,16,18],
    "S":[15,19,7,21,9]
},
index = ["Parker","William","Smith","Terry","Phill"])
info = info.reindex(["A","B","C","D","E"],fill_value=100)
print(info)'''

# reset_index() it is used to reset the index
'''info = pd.DataFrame([("Willaim","C"),("Smith","Java"),("Parker","Python"),("Phill",np.nan)],
index=['Hello','Hi','Welcome','Everyone'],
columns = ('name','language'))
info.reset_index(inplace=True)
print(info)'''

# set_index() is used to set the index.
'''info = pd.DataFrame({'Name':['Willaim','Phil','Paker','Smith'],
    'Age':[32,38,41,36],
    'id':[105,132,134,127]
})
info.set_index('Age',inplace = True)
print(info)'''

# Locate row (loc) Pandas use the loc attribute to return one or more specified rows.
'''data ={"calories":[40,380,390],"Duration":[50,45,47]}
myvar = pd.DataFrame(data)
print(myvar.loc[0,:])
print(myvar.loc[:,'Duration'])
print(myvar.loc[1,['calories','Duration']])
print(myvar.loc[[0,1]])'''

# read_csv() it used to read and perform operations using CSV files
'''df = pd.read_csv('data.csv')
print(df.to_string()) #It will return entire dataframe
print(df) # it will return first 5 and last5 rows
print(pd.options.display.max_rows) # to get the number of rows that is returned
pd.options.display.max_rows = 999 # this is used to increase the size of the entire dataframe.'''

#read_json() this is used to read the json data
'''df = pd.read_json('data1.json')
print(df.to_string())'''

'''df = pd.read_csv('data.csv')
print(df.head(10)) # First 10 rows
print(df.tail(10)) # last 10 rows
print(df.info()) # It gives more information about the dataframe'''

# dropna() to clean empty cells
'''data = pd.read_csv('data.csv')
newdf= data.dropna()
print(newdf.to_string())'''

# fillna() is used to fill empty cells with values.
'''data = pd.read_csv('data.csv')
data.fillna(130,inplace=True)
print(data.to_string())'''

# concat() is used to add two dataframes or Series
'''one = pd.DataFrame({'Name':['Parker','Phill','Smith'],
    'id':[108,119,127]
},index=['A','B','C'])
two= pd.DataFrame({'Name':['Terry','Jerry','Mery'],'id':[102,125,112]},index=['A','B','C'])
print(pd.concat([one,two]))'''

'''one = pd.DataFrame({'Name':['Parker','Phill','Smith'],
    'id':[108,119,127]
},index=['A','B','C'])
two= pd.DataFrame({'Name':['Terry','Jerry','Mery'],'id':[102,125,112]},index=['A','B','C'])
print(pd.concat([one,two],keys=['x','y']))'''

# Time Series to work with time series
'''info = pd.date_range('5/4/2013',periods=8,freq='S')
print(info)'''


































































































































































































































































































