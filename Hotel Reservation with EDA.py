#!/usr/bin/env python
# coding: utf-8

# ## Hotel Reservations Dataset
# ### Can you predict if customer is going to cancel the reservation ?

# The online hotel reservation channels have dramatically changed booking possibilities and customers’ behavior. A significant number of hotel reservations are called-off due to cancellations or no-shows. The typical reasons for cancellations include change of plans, scheduling conflicts, etc. This is often made easier by the option to do so free of charge or preferably at a low cost which is beneficial to hotel guests but it is a less desirable and possibly revenue-diminishing factor for hotels to deal with.
# 
# ### Can you predict if the customer is going to honor the reservation or cancel it ?

# #### Booking_ID: unique identifier of each booking
# 
# #### no_of_adults: Number of adults
# 
# #### no_of_children: Number of Children
# 
# #### no_of_weekend_nights: Number of weekend nights (Saturday or Sunday) the guest stayed or booked to stay at the hotel
# 
# #### no_of_week_nights: Number of week nights (Monday to Friday) the guest stayed or booked to stay at the hotel
# 
# #### type_of_meal_plan: Type of meal plan booked by the customer:
# 
# #### required_car_parking_space: Does the customer require a car parking space? (0 - No, 1- Yes)
# 
# #### room_type_reserved: Type of room reserved by the customer. The values are ciphered (encoded) by INN Hotels.
# 
# #### lead_time: Number of days between the date of booking and the arrival date
# 
# #### arrival_year: Year of arrival date
# 
# #### arrival_month: Month of arrival date
# 
# #### arrival_date: Date of the month
# 
# #### market_segment_type: Market segment designation.
# 
# #### repeated_guest: Is the customer a repeated guest? (0 - No, 1- Yes)
# 
# #### no_of_previous_cancellations: Number of previous bookings that were canceled by the customer prior to the current booking
# 
# #### no_of_previous_bookings_not_canceled: Number of previous bookings not canceled by the customer prior to the current booking
# 
# #### avg_price_per_room: Average price per day of the reservation; prices of the rooms are dynamic. (in euros)
# 
# #### no_of_special_requests: Total number of special requests made by the customer (e.g. high floor, view from the room, etc)
# 
# #### booking_status: Flag indicating if the booking was canceled or not.

# In[1]:


#Import necessary libraries
import numpy as np 
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import warnings
warnings.filterwarnings("ignore")
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


#importing Dataset
df = pd.read_csv("Hotel Reservations.csv")


# In[3]:


df.head()


# In[4]:


df.tail()


# In[5]:


df.shape


# In[6]:


df.info() #Dataframe information summary


# In[7]:


df.isnull().sum() #Count the number of missing values in each column of the dataframe


# In[8]:


df.columns #Print the column labels 


# In[9]:


df.describe()  #Generate summary statistics of the dataframe


# In[10]:


#Droping booking_id not useful 
df.drop("Booking_ID",axis=1,inplace=True)


# In[11]:


plt.figure(figsize=(50,50))
sns.heatmap(df.corr(),annot=True) #visualize the correlation matrix of numerical columns in the dataframe


# #### no_of_adults: Number of adults
# 

# In[12]:


df['no_of_adults'].unique()  #Get the unique values of the 'no_of_adults'


# In[13]:


df['no_of_adults'].value_counts()  #Count the number of occurrences of each unique value in the 'no_of_adults'


# In[14]:


px.histogram(df,x='no_of_adults',color="booking_status") #Create an interactive histogram using the Plotly Express library to visualize the distribution of the 'no_of_adults'


# ### It appears that the majority of bookings  have 2 adults.

# In[15]:


grouped_data = df.groupby('booking_status')['no_of_adults'].value_counts() #breakdown of the frequency distribution of 'no_of_adults' for each value of 'booking_status'


# In[16]:


# plot the data as a bar plot
fig, ax = plt.subplots(figsize=(10,6))
grouped_data.plot(kind='bar', ax=ax)

# add labels and title to the plot
ax.set_xlabel('Booking Status and Number of Adults')
ax.set_ylabel('Count')
ax.set_title('Counts of Booking Status and Number of Adults')
plt.show()


# #### For both categories, the most common value of 'no_of_adults' is 2, indicating that bookings with two adults are the most frequent.  

# In[17]:


import matplotlib.pyplot as plt

# group the data and count the values
grouped_data = df.groupby('booking_status')['no_of_adults'].value_counts().unstack()

# plot the data as a stacked bar chart
fig, ax = plt.subplots(figsize=(10,6))
grouped_data.plot(kind='bar', stacked=True, ax=ax)

# add labels and title to the plot
ax.set_xlabel('Number of Adults')
ax.set_ylabel('Count')
ax.set_title('Counts of Booking Status by Number of Adults')
plt.show()


# ``````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````
# 

# ### no_of_children: Number of Children

# In[18]:


df['no_of_children'].value_counts()  #Count the number of occurrences of each unique value in the no_of_children


# In[19]:


# sns.histplot(data=df, x="no_of_children")
px.histogram(df,x='no_of_children',color="booking_status")#visualize the distribution of the no_of_children


# #### The majority of bookings (33,577 out of 35,275) do not have any children. There are only a small number of bookings with more than 3 children, with 19 bookings having 3 children, 2 bookings having 9 children, and 1 booking having 10 children.

# In[20]:


groupby_children = df.groupby('booking_status')['no_of_children'].value_counts() #breakdown of the frequency distribution of 'Number of Children' for each value of 'booking_status'


# In[21]:


groupby_children


# In[22]:


# plot the data as a stacked bar chart
fig, ax = plt.subplots(figsize=(10,6))
groupby_children.plot(kind='bar', ax=ax)

# add labels and title to the plot
ax.set_xlabel('Number of children')
ax.set_ylabel('Count')
ax.set_title('Counts of Booking Status by Number of children')
plt.show()


# #### It can be seen that the most frequent value of 'no_of_children' for both categories combined is also 0, indicating that bookings without children are the most common. It is worth noting that there are very few bookings with more than 3 children in both categories.

# ``````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````

# ### Number of weekend nights (Saturday or Sunday) the guest stayed or booked to stay at the hotel

# In[23]:


df['no_of_weekend_nights'].unique() #Get the unique values of no_of_weekend_nights


# In[24]:


df['no_of_weekend_nights'].value_counts() #Count the number of occurrences of each unique value in the no_of_weekend_nights


# In[25]:


px.histogram(df,x='no_of_weekend_nights',color="booking_status") #visualize the distribution of the no_of_children


# In[26]:


df.groupby('no_of_weekend_nights')['booking_status'].value_counts()   #breakdown of the frequency distribution of 'no_of_weekend_nights' for each value of 'booking_status'


# In[27]:


groupby_weekend = df.groupby('no_of_weekend_nights')['booking_status'].value_counts() 


# In[28]:


# plot the data as a stacked bar chart
fig, ax = plt.subplots(figsize=(10,6))
groupby_weekend.plot(kind='bar', ax=ax)


# In[29]:


sns.barplot(x="booking_status",y="no_of_weekend_nights",data=df) 


# ``````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````

# ### no_of_week_nights: Number of week nights (Monday to Friday) the guest stayed or booked to stay at the hotel

# In[30]:


df['no_of_week_nights'].unique() #Get the unique values of no_of_week_nights


# In[31]:


df['no_of_week_nights'].value_counts() # Count the number of occurrences of each unique value in the no_of_week_nights


# In[32]:


px.histogram(df,x='no_of_week_nights',color='booking_status') #visualize the distribution of the no_of_week_nights


# #### The most common number of week nights is 2, which occurs for 11,444 bookings. The second and third most common numbers of week nights are 1 and 3, with 9,488 and 7,839 bookings, respectively. The least common values are 16, 17, and 13, which occur for only 2 or 3 bookings each.

# In[33]:


df.groupby('booking_status')['no_of_week_nights'].mean()  #The mean number of week nights for each booking status group


# In[34]:


sns.barplot(x="booking_status",y="no_of_week_nights",data=df) # barplot to compare the average number of week nights for each booking status


# ### Numbers of booking are Canceled more in weeks 

# ```````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````

# ### type_of_meal_plan: Type of meal plan booked by the customer:

# In[35]:


df['type_of_meal_plan'].unique() #Get the unique values of the type_of_meal_plan


# In[36]:


df['type_of_meal_plan'].value_counts()  #Count the number of occurrences of each unique value of the type_of_meal_plan


# In[37]:


px.histogram(df,x='type_of_meal_plan',color="booking_status") #visualize the distribution of the type_of_meal_plan


# ### The most common meal plan booked by the customers is Meal Plan 1

# In[38]:


groupby_meal =  df.groupby('type_of_meal_plan')['booking_status'].value_counts() #It shows the distribution of booking status for each type of meal plan.


# In[39]:


groupby_meal


# In[40]:


df = pd.get_dummies(df,columns=['type_of_meal_plan'])  #This code generates one-hot encoded dummy variables for the 'type_of_meal_plan' column 


# In[41]:


df.head()
pd.set_option('display.max_columns', None)


# In[42]:


df.head()


# ```````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````

# ### required_car_parking_space: Does the customer require a car parking space? (0 - No, 1- Yes)

# In[43]:


df['required_car_parking_space'].unique() #Get the unique values of required_car_parking_space


# In[44]:


df['required_car_parking_space'].value_counts() #Count the number of occurrences of each unique value


# In[45]:


sns.barplot(x="booking_status",y="required_car_parking_space",data=df) #visualize the distribution of the required_car_parking_space with booking_status


# In[46]:


groupby_required = df.groupby('required_car_parking_space')['booking_status'].value_counts()


# In[47]:


groupby_required


# In[48]:


# plot the data as a stacked bar chart
fig, ax = plt.subplots(figsize=(10,6))
groupby_required.plot(kind='bar', ax=ax)


# #### when people required_car_parking_space they are less likey to cancel booking

# ```````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````

#  ### room_type_reserved: Type of room reserved by the customer. The values are ciphered (encoded) by INN Hotels.

# In[49]:


df['room_type_reserved'].unique() #Get the unique values of room_type_reserved


# In[50]:


df['room_type_reserved'].value_counts() #Count the number of occurrences of each unique value of the room_type_reserved


# In[51]:


px.histogram(df,x='room_type_reserved',color="booking_status") #visualize the distribution of the room_type_reserved


# ### maximum number of people  book Room_type 1

# In[52]:


df.groupby('booking_status')['room_type_reserved'].value_counts() 


# In[53]:


import plotly.express as px

# group the data and count the values
grouped_data = df.groupby('booking_status')['room_type_reserved'].value_counts().reset_index(name='count')

# create the interactive bar chart
fig = px.bar(grouped_data, x='booking_status', y='count', color='room_type_reserved', barmode='group')

# add labels and title to the chart
fig.update_layout(title='Counts of Room Type Reserved by Booking Status')
fig.update_xaxes(title='Booking Status')
fig.update_yaxes(title='Count')

# display the chart
fig.show()


# In[54]:


df = pd.get_dummies(df,columns=['room_type_reserved'])  # This code generates one-hot encoded dummy variables for the room_type_reserved


# In[55]:


df.head()


# ```````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````

# ### lead_time: Number of days between the date of booking and the arrival date

# In[56]:


df['lead_time'].mean() #get mean time 


# In[57]:


df['lead_time'].median() #get median time 


# In[58]:


px.histogram(df,x='lead_time',color='booking_status') #visualize the distribution of the lead_time with booking_status


# ### from the above graph, we get to  know that when the lead time is less  the chances of canceling the booking are less 

# ``````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````

# ### arrival_year: Year of arrival date

# In[59]:


df['arrival_year'].value_counts() #Count the number of occurrences of each unique va


# In[60]:


px.histogram(df,x='arrival_year',color='booking_status') #visualize the distribution of the arrival_year with booking_status


# ``````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````

# ### arrival_month: Month of arrival date

# In[61]:


df['arrival_month'].value_counts() #Count the number of occurrences of each unique value


# In[62]:


px.histogram(df,x='arrival_month',color='booking_status') #visualize the distribution of the  arrival_month


# ### The highest number of bookings arrived in October, followed by September and August. The lowest number of bookings arrived in January.

# ``````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````

# In[63]:


df['arrival_date'].value_counts() ##Count the number of occurrences of each unique value


# In[64]:


px.histogram(df,x='arrival_date',color='booking_status') #visualize the distribution of the arrival_date with booking_status


# ``````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````

# ### market_segment_type: Market segment designation.

# In[65]:


df['market_segment_type'].unique() #Get the unique values of market_segment_type


# In[66]:


df['market_segment_type'].value_counts() #Count the number of occurrences of each unique value


# In[67]:


df.groupby('market_segment_type')['booking_status'].value_counts()


# In[68]:


px.histogram(df,x='market_segment_type',color='booking_status')  #visualize the distribution of the market_segment_type


# ### Complementary type never canceled there booking. Chances of canceling is more in online types

# In[69]:


df = pd.get_dummies(df,columns=['market_segment_type']) #This code generates one-hot encoded dummy variables for the market_segment_type


# In[70]:


df.head()


# ``````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````

# ### repeated_guest: Is the customer a repeated guest? (0 - No, 1- Yes)

# In[71]:


df['repeated_guest'].value_counts() #Count the number of occurrences of each unique value of repeated_guest 


# In[72]:


df.groupby("repeated_guest")['booking_status'].value_counts() 


# In[73]:


px.histogram(df,x='repeated_guest',color='booking_status') #visualize the distribution of the


# ### Repeated guest are very less 

# ``````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````

# ### no_of_previous_cancellations: Number of previous bookings that were canceled by the customer prior to the current booking

# In[74]:


df['no_of_previous_cancellations'].unique()        


# In[75]:


df['no_of_previous_cancellations'].value_counts()


# In[76]:


sns.histplot(df,x='no_of_previous_cancellations')


# ### majority of people have no history of previous_cancellations

# ``````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````

# ### avg_price_per_room: Average price per day of the reservation; prices of the rooms are dynamic. (in euros)

# In[77]:


df['avg_price_per_room'].value_counts()  #Count the number of occurrences of each unique value avg_price_per_room           


# In[78]:


df['avg_price_per_room'].median() #get median value


# In[79]:


df['avg_price_per_room'].mode() #get mode value 


# In[80]:


px.histogram(df,x='avg_price_per_room',color='booking_status')  #visualize the distribution of the avg_price_per_room


# ``````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````
# 

# ###  no_of_special_requests: Total number of special requests made by the customer (e.g. high floor, view from the room, etc)

# In[81]:


df['no_of_special_requests'].unique()  #Get the unique values of no_of_special_requests


# In[82]:


df['no_of_special_requests'].value_counts() # Count the number of occurrences of each unique values of no_of_special_requests


# In[83]:


df.groupby('no_of_special_requests')['booking_status'].value_counts()


# In[84]:


px.histogram(df,x='no_of_special_requests',color='booking_status') #visualize the distribution of the no_of_special_requests


# ### People who make more special requests tend to not cancel their booking.

# ``````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````

# ### booking_status: Flag indicating if the booking was canceled or not.

# In[85]:


df['booking_status'].value_counts()


# In[86]:


px.pie(df,'booking_status') #Pie chart of booking status


# ### In our data set 67.2% people have not Canceled the booking and 32.8% people have Canceled the booking

# In[87]:


# mapping not canceled as 1 and canceled as 0
df['booking_status'] = df['booking_status'].map({'Not_Canceled':1,'Canceled':0})


# In[88]:


df.head()


# ### Dropping columns 

# In[89]:


df = df.drop("arrival_year",axis=1)


# In[90]:


df = df.drop("arrival_date",axis=1)


# In[91]:


df = df.drop(['no_of_previous_cancellations','no_of_previous_bookings_not_canceled','no_of_week_nights','no_of_children','required_car_parking_space'],axis=1)


# In[92]:


df.head()


# In[93]:


x = df.drop("booking_status",axis=1)
y = df['booking_status']


# In[94]:


x.head()


# In[95]:


y.head()


# ## Train Test split

# In[96]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2,stratify=y, random_state=50)


# ##  Logistics regression

# In[97]:


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,recall_score,classification_report,confusion_matrix

LR = LogisticRegression()


# ### Train Logistics

# In[98]:


LR.fit(X_train,y_train)


# In[99]:


y_train_pred_LR = LR.predict(X_train)
y_test_pred_LR = LR.predict(X_test)


# ### Training accuracy

# In[100]:


accuracy_score(y_train,y_train_pred_LR)


# ### Testing accuracy

# In[101]:


accuracy_score(y_test,y_test_pred_LR)


# In[102]:


prob = LR.predict_proba(X_test)


# In[103]:


prob1 =prob[:,0]


# In[104]:


from sklearn.metrics import roc_auc_score,roc_curve
tpr,fpr,threshold=roc_curve(y_test,prob1)


# In[105]:


auc = roc_auc_score(y_test,prob1)


# In[106]:


plt.plot(fpr,tpr,label=str(auc))
plt.legend()


# ### AUC Curve

# ###  Classification report

# In[107]:


print(classification_report(y_test,y_test_pred_LR))


# In[108]:


confusion_matrix(y_test, y_test_pred_LR)


# ###  Decision tree classifier

# In[109]:


from sklearn.tree import DecisionTreeClassifier


# In[110]:


dc = DecisionTreeClassifier(criterion='gini', splitter='best', max_depth=8, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features=None, random_state=None, max_leaf_nodes=None, min_impurity_decrease=0.0, class_weight=None, ccp_alpha=0.0)


# ### Train Decision tree classifier 

# In[111]:


dc.fit(X_train,y_train)


# In[112]:


y_train_pred_dc =dc.predict(X_train)
y_test_pred_dc = dc.predict(X_test)


# ###  Training accuracy

# In[113]:


accuracy_score(y_train,y_train_pred_dc)


# ### Test accuracy 

# In[114]:


accuracy_score(y_test, y_test_pred_dc)


# In[115]:


confusion_matrix(y_test,y_test_pred_dc)


# In[116]:


print(classification_report(y_test,y_test_pred_dc))


# In[ ]:





# ### Random Forest Classifier

# In[117]:


from sklearn.ensemble import RandomForestClassifier


# In[118]:


RD = RandomForestClassifier(max_depth=10)


# ### Train Random Forest Classifier

# In[119]:


RD.fit(X_train,y_train)


# In[120]:


y_train_pred_rd = RD.predict(X_train)
y_test_pred_rd  = RD.predict(X_test)


# ###  Training accuracy

# In[121]:


accuracy_score(y_train,y_train_pred_rd)


# ### Test accuracy 

# In[122]:


accuracy_score(y_test,y_test_pred_rd)


# In[123]:


print(classification_report(y_test,y_test_pred_rd))


# In[124]:


confusion_matrix(y_test,y_test_pred_rd)


# In[125]:


accuracy = pd.DataFrame([(0.7991729841488628 , 0.7935217091660923) , (0.8662301860785665 , 0.8618883528600965) ,
(0.87170916609235 ,  0.8657477601654032 )],index=('Logistics','Decision Tree','Random forest'),columns=['Train','Test'])


# In[126]:


accuracy


# # Based on the evaluation metrics, the Random Forest model achieved the highest accuracy, indicating an 86% probability of correctly predicting whether a customer will cancel their reservation or not. Therefore, we can consider using this model for prediction purposes.
