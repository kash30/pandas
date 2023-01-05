#!/usr/bin/env python
# coding: utf-8

# # DS/CMPSC 410 MiniProject Deliverable #1 
# 
# # Fall 2022
# ## Instructor: Prof. John Yen
# ## TA: Rupesh Prajapati, Haiwen Guan
# ## LA: Zining Yin
# ## Learning Objectives
# - Be able to identify frequent 2 port sets and 3 port sets that are scanned by scanners in the Darknet dataset
# - Be able to adapt the Aprior algorithm by incorporating DFS, suitable threshold, and pruning strategies.
# - Be able to improve the performance of frequent port set mining by suitable reuse of RDD, together with appropriate persist and unpersist on the reused RDD.
# - After successful execution in the local mode, modify the code for cluster mode, obtain run-time information (in log file), and final frequent 2-port sets and 3-port sets using the big Darknet dataset (`Day_2020_profile.csv`).
# 
# ### Items to submit:
# - Completed Jupyter Notebook (using small Darknet dataset `sampled_profile.csv`) in HTML format.
# - .py file for mining 2 port sets and 3 port sets in cluster mode using the big Darknet dataset `Day_2020_profile.csv`.
# - log file for a successful execution in the cluster mode, including its run-time performance.
# - The csv file of frequent 2-port sets generated in the CLUSTER mode.
# - The csv file of frequent 3-port sets generated in the CLUSTER mode.
# 
# ### Total points: 100 
# - Exercise 1: 5 points
# - Exercise 2: 10 points
# - Exercise 3: 5 points
# - Exercise 4: 10 points
# - Exercise 5: 20 points
# - Exercise 6: 10 points
# - Exercise 7: 40 points
#   
# ### Due: midnight, November 6, 2022
# ### Early Submission Bonus: 10 points for submission before midnight November 4

# In[1]:


import pyspark
import csv
import pandas as pd


# In[2]:


from pyspark import SparkContext
from pyspark.sql import SparkSession
from pyspark.sql.types import StructField, StructType, StringType, LongType
from pyspark.sql.functions import col, column
from pyspark.sql.functions import expr
from pyspark.sql.functions import split
from pyspark.sql import Row
from pyspark.ml import Pipeline
from pyspark.ml.feature import OneHotEncoder, StringIndexer, VectorAssembler, IndexToString
from pyspark.ml.clustering import KMeans


# In[3]:


ss = SparkSession.builder.appName("Mini Project #1 Freqent Port Sets").getOrCreate()


# # Exercise 1 (5 points)
# - Complete the path below for reading "sampled_profile.csv" you downloaded from Canvas, uploaded to your Mini Project 1 folder. 
# - Fill in your Name : Kashish Gujral

# In[4]:


# In the cluster mode, change this line to
Scanners_df = ss.read.csv("/storage/home/kmg6272/work/MiniProj1/Day_2020_profile.csv", header= True, inferSchema=True )
#Scanners_df = ss.read.csv("/storage/home/kmg6272/work/MiniProj1/sampled_profile.csv", header=True, inferSchema=True)


# ## We can use printSchema() to display the schema of the DataFrame Scanners_df to see whether it was inferred correctly.

# In[5]:


Scanners_df.printSchema()


# ## Overview: Part A, B, and C uses the small dataset sampled_profile.csv. Part D (cluster mode) uses the large dataset Day_2020_profile.csv
# - Part A: Transform the feature "ports_scanned_str" into an array/list of ports scanned by each scanner.
# - Part B: Finding frequent ports based on a threshold.
# - Part C: Finding frequent 2-port sets and 3-port sets based on the threshold (in local mode)
# - Part D (cluster mode): Finding frequent 2-port sets and 3-port sets from the large dataset.

# # Part A Transfosrm the feature "ports_scanned_str" into an array of ports.
# ### The original value of the column is a string that connects all the ports scanned by a scanner. The different ports that are open by a scanner are connected by dash "-". For example, "81-161-2000" indicates the scanner has scanned three ports: port 81, port 161, and port 2000. Therefore, we want to use split to separate it into an array of ports by each scanner.  This transformation is important because it enables the identification of frequent ports scanned by scanners.

# ## The original value of the column "ports_scanned_str" 

# In[6]:


#Scanners_df.select("ports_scanned_str").show(10)


# ## Convert the Column 'ports_scanned_str' into an Array of ports scanned by each scanner (row)

# In[7]:


Scanners_df2=Scanners_df.withColumn("Ports_Array", split(col("ports_scanned_str"), "-") )


# In[8]:


#Scanners_df2.show(3)


# ## For Mining Frequent Port Sets being scanned, we only need the column ```Ports_Array```

# In[9]:


Ports_Scanned_RDD = Scanners_df2.select("Ports_Array").rdd


# In[10]:


#Ports_Scanned_RDD.take(5)


# # Convert an RDD of Ports_Array into an RDD of port list.
# ## Because the first element of the Row object in the RDD is the Ports_Array, we can access the content of Ports_Array using index 0.
# ## The created RDD `multi_Ports_list_RDD` will be used in finding frequent port sets below (Part B, C, D).

# In[11]:


multi_Ports_list_RDD = Ports_Scanned_RDD.map(lambda x: x[0])


# In[12]:


#multi_Ports_list_RDD.take(5)


# # Part B: Finding all ports that have been scanned by more than 999 scanners in local mode, (more than 4999 in the cluster mode).

# ## Because each port number in the Ports_Array column for each row/scanner occurs only once, we can count the total occurance of each port number through flatMap.

# ## `flatMap` flatten the RDD into a list of ports. We can then just count the occurance of each port in the RDD, which is the number of scanners that scan the port.   

# # Exercise 2 (10 points) Complete the code below to calculate the total number of scanners that scan each port. 

# In[13]:


#multi_Ports_list_RDD.take(3)


# In[14]:


port_list_RDD = multi_Ports_list_RDD.flatMap(lambda x: x )


# In[15]:


#port_list_RDD.take(5)


# In[18]:


Port_count_RDD = port_list_RDD.map(lambda x: (x,1) )
#Port_count_RDD.take(5)


# In[19]:


Port_count_total_RDD = Port_count_RDD.reduceByKey(lambda x,y: x+y)
#Port_count_total_RDD.take(5)


# ## How many ports are being scanned?

# In[20]:


Port_count_total_RDD.count()


# # To sort port by their frequency count, we flip the order between port and count so that count is the key, then we apply sortByKey

# In[21]:


Sorted_Count_Port_RDD = Port_count_total_RDD.map(lambda x: (x[1], x[0])).sortByKey( ascending = False)


# In[22]:


#Sorted_Count_Port_RDD.take(10)


# ## Since we are interested in ports that are scanned by more than 999 scanners (in the local mode), we use 999 as the threshold.
# ## In the cluster mode, we need to change the thresdhold to 4999.
# ## Exercise 3 (5 points) Complete the following code to filter for ports that have more than 999 scanners.

# In[23]:


threshold = 4999
# This threshold value is for the local mode.  You need to change it to 4999 for the cluster mode.
Filtered_Port_RDD= Sorted_Count_Port_RDD.filter(lambda x: x[0] > threshold )
total_freq_port_no = Filtered_Port_RDD.count()


# In[24]:


#Filtered_Port_RDD.take(5)


# In[25]:


#print(total_freq_port_no)


# # We want to get a list of frequent 1-port set to be used in finding frequent 2-port set and frequent 3-port sets. 
# ## Because ports are the second element in the key-value tuple, we access it by mapping `(lambda x: x[1])` to the `Filtered_Port_RDD` generated above.
# ## After we apply collect to the mapped RDD, we get a list of single ports that are scanned by at least 1000 scanners in the small dataset (or at least 5000 scanners in the large dataset).

# In[26]:


Top_Ports = Filtered_Port_RDD.map(lambda x: x[1]).collect()


# In[27]:


#print(Top_Ports)


# In[28]:


Top_1_Port_count = len(Top_Ports)


# In[29]:


#print(Top_1_Port_count)


# # Part C Finding Frequent Two-Port Sets and Three-Port Sets 

# # Pruning Strategy: Because we do not need to consider any scanners that scan only 1 port in finding frequent 2-port sets or frequent 3-port sets, we can filter `multi_Ports_list_RDD` to remove those single port scanners.

# In[30]:


#multi_Ports_list_RDD.take(5)


# In[31]:


#multi_Ports_list_RDD.count()


# ## We can use the Python function `len` to filter for scanners that scan more than one port.
# - Below are examples of applying `len` to a list.

# In[32]:


list1 = ['17128', '17136']
list2 = ['22']


# In[33]:


len(list1)


# In[34]:


len(list2)


# # Exercise 4 (10 points) 
# - (a) Complete the code below to filter for scanners that scan more than one port. (5 points)
# - (b) Enter in the Exercise 3 Answer cell below the reduction of scanners due to this pruning strategy. (5 points)

# In[35]:


multi_ports_scanners_RDD = multi_Ports_list_RDD.filter(lambda x: len(x) > 1 )


# In[36]:


#multi_ports_scanners_RDD.take(5)


# In[37]:


#multi_ports_scanners_RDD.count()


# # We will use `multi_ports_scanners_RDD` in the reamining code for finding frequent 2-port sets and frequent 3 port sets.

# # Answer to Exercise 4 (b)
# - What is the reduction of scanners based on this pruning strategy? 
# - Your Answer: 73663

# ## The following two code cells demonstrate how we use Python `in` test for list to filter for scanners who scan one or more specific ports, then count the number of scanners that satisfy that criteria.

# In[38]:


count_17132_23 = multi_ports_scanners_RDD.filter(lambda x: ('17132' in x) and ('23' in x)).count()


# In[39]:


#print(count_17132_23)


# In[40]:


count_17132_445 = multi_ports_scanners_RDD.filter(lambda x: ('17132' in x) and ('445' in x)).count()


# In[41]:


#print(count_17132_445)


# # Since we will be using `multi_ports_scanners_RDD` in the reamining code for finding frequent 2-port sets and frequent 3 port sets, we display the content of a few RDD to double check that we do not see any 1-port scanners in the RDD.

# In[42]:


#multi_ports_scanners_RDD.take(5)


# In[43]:


#print(Top_Ports)


# ## As mentioned earlier, to check whether a scanner scans a specific port (e.g., the first in the Top_Ports list: port 17128), we can use python code such as `(lambda x: Top_Ports[0] in x)` to filter for scanners that scan the sepcific port.

# ## We will use this lambda expression for filtering the `multi_ports_scanners_RDD` for port list that contains a specific Top port.

# In[44]:


Top_1_Port_count = len(Top_Ports)


# In[45]:


#print(Top_1_Port_count)


# # Part D: Finding Frequent 2-Port Sets and 3-Port Sets

# # Exercise 5 (20 points)  Complete the following code (including suitabler persist and unpersist) to find BOTH frequent 2 port sets AND frequent 3 port sets 
# - Hints:
# -- Frequent two port sets are saved in Pandas dataframe `Two_Port_Sets_df` 
# -- Frequent three port sets are saved in Pandas dataframe `Three_Port_sets_df`
# -- Use two `index` variables: `index2` for `Two_Port_Sets_df`, and `index3` for `Three_Port_sets_df`.  

# In[46]:


# Initialize a Pandas DataFrame to store frequent port sets and their counts 
Two_Port_Sets_df = pd.DataFrame( columns = ['Port Sets', 'count'])
Three_Port_Sets_df = pd.DataFrame( columns= ['Port Sets', 'count'])
# Initialize the index to Two_Port_Sets_df
index2 = 0
# Initialize the index to Three_Port_Sets_df
index3 = 0
# Set the threshold for Frequent Port Sets to be 1000 in local mode, 5000 in cluster mode.
# This threshold needs to be changed to 4999 in the cluster mode.
threshold = 4999
multi_ports_scanners_RDD.persist()
for i in range(0, Top_1_Port_count-1):
    Scanners_port_i_RDD = multi_ports_scanners_RDD.filter(lambda x: Top_Ports[i] in x)
    Scanners_port_i_RDD.persist()  
    # We do not need to filter for threshold for 1-port sets because all ports in Top_Ports have a
    # frequency higher than the threshold.
    for j in range(i+1, Top_1_Port_count-1):
        Scanners_port_i_j_RDD = Scanners_port_i_RDD.filter(lambda x: Top_Ports[j] in x)
        Scanners_port_i_j_RDD.persist()
        two_ports_count = Scanners_port_i_j_RDD.count()
        if two_ports_count > threshold:
            Two_Port_Sets_df.loc[index2] = [ [Top_Ports[i], Top_Ports[j]], two_ports_count] 
            index2 = index2 + 1
            # The print statement is for running in the local mode.  It can be commented out for running in the cluster mode.
            print("Two Ports:", Top_Ports[i], " , ", Top_Ports[j], ", Count: ", two_ports_count)
            for k in range(j+1, Top_1_Port_count -1):
                Scanners_port_i_j_k_RDD = Scanners_port_i_j_RDD.filter(lambda x: Top_Ports[k] in x)
                three_ports_count = Scanners_port_i_j_k_RDD.count()
                if three_ports_count > threshold:
                    Three_Port_Sets_df.loc[index3] = [ [Top_Ports[i], Top_Ports[j], Top_Ports[k]], three_ports_count]
                    index3 = index3 + 1
#                   print("Three Ports: ", Top_Ports[i], ", ", Top_Ports[j], ",  ", Top_Ports[k], ": Count ", three_ports_count)
        Scanners_port_i_j_RDD.unpersist()
    Scanners_port_i_RDD.unpersist()


# # Save the Pandas dataframes of frequent 2-port sets and frequent 3-port sets in CSV files

# # Exercise 6 (10 points)
# Complete the following code to save your frequent 2 port sets and 3 port sets in an output file.

# In[48]:


# These output file names need to be changed in the cluster mode, so that you can compare them with those from the local mode.
output_path_2 = "/storage/home/kmg6272/work/MiniProj1/MiniProj1_2Ports.csv"
output_path_3 = "/storage/home/kmg6272/work/MiniProj1/MiniProj1_3Ports.csv"
Two_Port_Sets_df.to_csv(output_path_2)
Three_Port_Sets_df.to_csv(output_path_3)


# In[46]:


ss.stop()


# # Part D (cluster mode): Finding frequent 2-port sets and 3-port sets from the large dataset.

# # Exercise 7 (40 points)
# - Remove .master("local") from SparkSession statement
# - Change the input file to "Day_2020_profile.csv"
# - Change the threshold from 999 to 4999.
# - Change the output files to two different directories from the ones you used in local mode.
# - Export the notebook as a .py file
# - Run spark-submit on ICDS Roar 
# - Submit the following items:
# -- (a) the .py file for cluster mode (10%)
# -- (b) Two output csv file for frequent 2-port sets and for frequent 3-port sets generated in the cluster mode. (10%)
# -- (c) The log file that shows the run-time information. (10%)
# -- (d) Discuss (in the cell below) three things you noticed that are interesting/surprising from the frequent 3-port sets (10%)

# # Your Answer to Exercise 7 (d):
# Type your answer here.
