#!/usr/bin/env python
# coding: utf-8

# # DS/CMPSC 410 MiniProject Deliverable #2
# 
# # Fall 2022
# ### Instructor: Prof. John Yen
# ### TA: Rupesh Prajapati and Haiwen Guan
# ### LA: Zining Yin
# 
# ### Learning Objectives
# - Be able to apply k-means clustering to the Darknet dataset.
# - Be able to identify the set of top k ports for one-hot encoding ports scanned.
# - Be able to characterize generated clusters using cluster centers.
# - Be able to compare and evaluate the result of k-means clustering with different features using Silhouette score and external labels.
# - After successful clustering of the small Darknet dataset, conduct clustering on the large Darknet dataset.
# 
# ### Total points: 100 
# - Exercise 1: 5 points
# - Exercise 2: 5 points 
# - Exercise 3: 10 points 
# - Exercise 4: 5 points
# - Exercise 5: 10 points
# - Exercise 6: 10 points
# - Exercise 7: 15 points
# - Exercise 8: 40 points
#   
# ### Due: 11:59 pm, November 13, 2022
# ### Early Submission bonus (before midnight November 11th): 10 points

# In[1]:


import pyspark
import csv


# In[2]:


from pyspark import SparkContext
from pyspark.sql import SparkSession
from pyspark.sql.types import StructField, StructType, StringType, LongType
from pyspark.sql.functions import col, column
from pyspark.sql.functions import expr
from pyspark.sql.functions import split
from pyspark.sql.functions import array_contains
from pyspark.sql import Row
from pyspark.ml import Pipeline
from pyspark.ml.feature import OneHotEncoder, StringIndexer, VectorAssembler, IndexToString
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator


# In[3]:


ss = SparkSession.builder.master("local").appName("MiniProject 2 k-meas Clustering using OHE").getOrCreate()


# ## Exercise 1 (5 points)
# Complete the path for input file in the code below and enter your name in this Markdown cell:
# - Name: Kashish Gujral
# ### Note: You will need to change the name of the input file in the cluster mode to `Day_2020_profile.csv`

# In[4]:


Scanners_df = ss.read.csv("/storage/home/kmg6272/work/MiniProj2/sampled_profile.csv", header= True, inferSchema=True )


# ## We can use printSchema() to display the schema of the DataFrame Scanners_df to see whether it was inferred correctly.

# In[5]:


Scanners_df.printSchema()


# # In this lab, our goal is to answer the question:
# ## Q: What groups of scanners are similar in the ports they scan?
# 
# ### Because we know (from MiniProject 1) about two third of the scanners scan only 1 port, we can separate them from the other scanners.

# ### Because the feature `numports` record the total number of ports being scanned by each scanner, we can use it to separate 1-port-scanners from multi-port-scanners.

# In[6]:


one_port_scanners = Scanners_df.where(col('numports') == 1)


# In[7]:


one_port_scanners.show(3)


# In[8]:


multi_port_scanners = Scanners_df.where(col("numports") > 1)


# In[9]:


multi_port_scanners_count = multi_port_scanners.count()


# In[10]:


print(multi_port_scanners_count)


# In[11]:


ScannersCount_byNumPorts = multi_port_scanners.groupby("numports").count()


# In[12]:


ScannersCount_byNumPorts.show(3)


# In[13]:


SortedScannersCount_byNumPorts= ScannersCount_byNumPorts.orderBy("count", ascending=False)


# In[15]:


output1 = "/storage/home/kmg6272/work/MiniProj2/small/SortedScannersCountbyNumPorts.csv"
SortedScannersCount_byNumPorts.write.csv(output1)


# In[16]:


ScannersCount_byNumPorts.where(col("count")==1).show(10)


# # We noticed that some of the scanners that scan for very large number of ports (we call them Extreme Scanners) is unique in the number of ports they scan.
# ## A heuristic to separate extreme scanners: Find the largest number of ports that are scanned by at least two scanners. Use the number as the threshold to filter extreme scanners.

# In[17]:


non_rare_NumPorts = SortedScannersCount_byNumPorts.where(col("count") > 1)


# In[18]:


non_rare_NumPorts.show(3)


# In[19]:


max_non_rare_NumPorts_df = non_rare_NumPorts.agg({"numports" : "max"})
max_non_rare_NumPorts_df.show()


# In[20]:


max_non_rare_NumPorts_rdd = max_non_rare_NumPorts_df.rdd.map(lambda x: x[0])
max_non_rare_NumPorts_rdd.take(1)


# In[21]:


max_non_rare_NumPorts_list = max_non_rare_NumPorts_rdd.collect()
print(max_non_rare_NumPorts_list)


# In[22]:


max_non_rare_NumPorts=max_non_rare_NumPorts_list[0]
print(max_non_rare_NumPorts)


# ## We are going to focus on the grouping of scanners that scan at least two ports, and do not scan very large number of ports. We will call these scanners Non-extreme Multi-port Scanners.
# ## We will save the extreme scanners in a csv file so that we can process it separately.

# In[23]:


extreme_scanners = Scanners_df.where(col("numports") > max_non_rare_NumPorts)


# In[25]:


path2="/storage/home/kmg6272/work/MiniProj2/small/ExtremeScanners.csv"
extreme_scanners.write.option("header",True).csv(path2)


# In[26]:


non_extreme_multi_port_scanners = Scanners_df.where(col("numports") <= max_non_rare_NumPorts).where(col("numports") > 1)


# In[27]:


non_extreme_multi_port_scanners.count()


# # Part A: One Hot Encoding of Top 100 Ports
# We want to apply one hot encoding to the top 100 ports scanned by scanners. 
# - A1: Find top k ports scanned by non_extreme_multi_port scanners (This is similar to the first part of MiniProject 1)
# - A2: Generate One Hot Encodding for these top k ports

# In[28]:


non_extreme_multi_port_scanners.select("ports_scanned_str").show(4)


# # For each port scanned, count the Total Number of Scanners that Scan the Given Port
# Like MiniProject 1, to calculate this, we need to 
# - (a) convert the ports_scanned_str into an array/list of ports
# - (b) Convert the DataFrame into an RDD
# - (c) Use flatMap to count the total number of scanners for each port.

# # The Following Code Implements the three steps.
# ## (a) Split the column "Ports_Array" into an Array of ports.

# In[29]:


# (a)
NEMP_Scanners_df=non_extreme_multi_port_scanners.withColumn("Ports_Array", split(col("ports_scanned_str"), "-") )
NEMP_Scanners_df.show(10)


# ## (b) We convert the column ```Ports_Array``` into an RDD so that we can apply flatMap for counting.

# In[30]:


Ports_Scanned_RDD = NEMP_Scanners_df.select("Ports_Array").rdd


# In[31]:


Ports_Scanned_RDD.take(5)


# ## (c) Because each port number in the Ports_Array column for each row/scanner occurs only once, we can count the total number of scanners by counting the total occurance of each port number through flatMap.
# ### Because each element of the rdd is a Row object, we need to first extract the first element of the row object, which is the list of Ports, from the ``Ports_Scanned_RDD``
# ### We can then count the total number of occurance of a port using map and reduceByKey, like counting word/hashtag frequency in tweets.

# In[32]:


Ports_Scanned_RDD.take(3)


# In[33]:


Ports_list_RDD = Ports_Scanned_RDD.map(lambda row: row[0] )


# In[34]:


Ports_list_RDD.take(3)


# In[35]:


Ports_list2_RDD = Ports_list_RDD.flatMap(lambda x: x )


# In[36]:


Ports_list2_RDD.take(7)


# In[37]:


Port_count_RDD = Ports_list2_RDD.map(lambda x: (x, 1))
Port_count_RDD.take(7)


# In[38]:


Port_count_total_RDD = Port_count_RDD.reduceByKey(lambda x,y: x+y)
Port_count_total_RDD.take(5)


# # Exercise 2 (5%)
# ### Complete the code below to find the total number of ports being scanned by non-extreme multi-port scanners

# In[39]:


Port_count_total_RDD.count()


# ## Note: These are the total number of ports scanned by Non-extreme Multi-port scanners in the small dataset.  Is the result what you expected?

# ### The code below finds top k ports scanned by non-extreme multi-port scanners
# ### We will set k to 100 for mini-project 2.

# In[40]:


Sorted_Count_Port_RDD = Port_count_total_RDD.map(lambda x: (x[1], x[0])).sortByKey( ascending = False)


# In[41]:


Sorted_Count_Port_RDD.take(100)


# In[42]:


top_ports= 100
Sorted_Ports_RDD= Sorted_Count_Port_RDD.map(lambda x: x[1] )
Top_Ports_list = Sorted_Ports_RDD.take(top_ports)


# In[43]:


Top_Ports_list


# #  A.2 One Hot Encoding of Top K Ports
# ## One-Hot-Encoded Feature/Column Name
# Because we need to create a name for each one-hot-encoded feature, which is one of the top k ports, we can adopt the convention that the column name is "PortXXXX", where "XXXX" is a port number. This can be done by concatenating two strings using ``+``.

# In[44]:


Top_Ports_list[0]


# In[45]:


FeatureName = "Port"+Top_Ports_list[0]


# In[46]:


FeatureName


# ## One-Hot-Encoding using withColumn and array_contains

# # Exercise 3 (10 points) Complete the code below for One-Hot-Encoding of the first top port.

# In[47]:


from pyspark.sql.functions import array_contains


# In[48]:


NEMP_Scanners2_df= NEMP_Scanners_df.withColumn("Port"+Top_Ports_list[0], array_contains("Ports_Array", Top_Ports_list[0]))


# In[49]:


NEMP_Scanners2_df.show(10)


# ## Verify the Correctness of One-Hot-Encoded Feature
# ## Exercise 4 (5 points)
# ### Check whether one-hot encoding of the first top port is encoded correctly by completing the code below and enter your answer the in the next Markdown cell.

# In[52]:


First_top_port_scanners_count = NEMP_Scanners2_df.where(col("Port17132") == True).rdd.count()


# In[53]:


Sorted_Count_Port_RDD.count()


# In[54]:


print(First_top_port_scanners_count)


# ## Answer for Exercise 4:
# - The total number of scanners that scan the first top port, based on ``Sorted_Count_Port_RDD`` is:
# - Is this number the same as the number of scanners whose One-Hot-Encoded feature of the first top port is True?

# ## Generate Hot-One Encoded Feature for each of the top k ports in the Top_Ports_list
# 
# - Iterate through the Top_Ports_list so that each top port is one-hot encoded into the DataFrame for non-extreme multi-port scanners (i.e., `NEMP_Scanners2.df`).

# ## Exercise 5 (10 points)
# Complete the following PySpark code for encoding the n ports using One Hot Encoding, where n is specified by the variable ```top_ports```

# In[55]:


top_ports


# In[56]:


Top_Ports_list[top_ports - 1]


# In[57]:


for i in range(0, top_ports):
    # "Port" + Top_Ports_list[i]  is the name of each new feature created through One Hot Encoding Top_Ports_list
    NEMP_Scanners3_df = NEMP_Scanners2_df.withColumn("Port" + Top_Ports_list[i], array_contains("Ports_Array",Top_Ports_list[i]))
    NEMP_Scanners2_df = NEMP_Scanners3_df


# In[58]:


NEMP_Scanners2_df.printSchema()


# ## Exercise 6 (10 points)
# Complete the code below to use k-means to cluster non-extreme multi-port scanners using one-hot-encoded top 100 ports.

# ## Specify Parameters for k Means Clustering

# In[59]:


input_features = [ ]
for i in range(0, top_ports ):
    input_features.append( "Port"+ Top_Ports_list[i] )


# In[60]:


print(input_features)


# In[61]:


va = VectorAssembler().setInputCols(input_features).setOutputCol("features")


# In[63]:


data= va.transform(NEMP_Scanners2_df)


# In[64]:


data.show(3)


# In[65]:


data.persist()


# In[66]:


km = KMeans(featuresCol= "features", predictionCol="prediction").setK(200).setSeed(123)
km.explainParams()


# In[67]:


kmModel=km.fit(data)


# In[68]:


kmModel


# In[69]:


predictions = kmModel.transform(data)


# In[70]:


predictions.show(3)


# In[71]:


Cluster1_df=predictions.where(col("prediction")==0)


# In[72]:


Cluster1_df.count()


# In[73]:


summary = kmModel.summary


# In[74]:


summary.clusterSizes


# In[75]:


evaluator = ClusteringEvaluator()
silhouette = evaluator.evaluate(predictions)


# In[76]:


print('Silhouette Score of the Clustering Result is ', silhouette)


# In[77]:


centers = kmModel.clusterCenters()


# In[78]:


print(centers)


# # Record cluster index, cluster size, percentage of Mirai scanners, and cluster centers for each clusters formed.
# ## The value of cluster center for a OHE top port is the percentage of data/clusters in the cluster that scans the top port. For example, a cluster center `[0.094, 0.8, 0, ...]` indicates the following
# - 9.4% of the scanners in the cluster scan Top_Ports_list[0]: port 17132
# - 80% of the scanners in the cluster scan Top_Ports_list[1]: port 17130
# - No scanners in the cluster scan Top_Ports_list[2]: port 17140

# # Exercise 7 (15 points) Complete the code below for computing the percentage of Mirai scanners for each scanner, and record it together with cluster centers for each cluster.

# In[79]:


import pandas as pd
import numpy as np
import math


# In[80]:


# Define columns of the Pandas dataframe
column_list = ['cluster ID', 'size', 'mirai_ratio' ]
for feature in input_features:
    column_list.append(feature)
clusters_summary_df = pd.DataFrame( columns = column_list )
for i in range(0, 200):
    cluster_i = predictions.where(col('prediction')==i)
    cluster_i_size = cluster_i.count()
    cluster_i_mirai_count = cluster_i.where(col('mirai')).count()
    cluster_i_mirai_ratio = cluster_i_mirai_count/cluster_i_size
    if cluster_i_mirai_count > 0:
        print("Cluster ", i, "; Mirai Ratio:", cluster_i_mirai_ratio, "; Cluster Size: ", cluster_i_size)
    cluster_row = [i, cluster_i_size, cluster_i_mirai_ratio]
    for j in range(0, len(input_features)):
        cluster_row.append(centers[i][j])
    clusters_summary_df.loc[i]= cluster_row


# In[81]:


path4= "/storage/home/kmg6272/work/MiniProj2/small/Clusters_Summary_100OHE_k200.csv"
clusters_summary_df.to_csv(path4, header=True)


# # Exercise 8 (40 points)
# Modify the Jupyter Notebook for running in cluster mode using the big dataset (Day_2020_profile.csv). Make sure you change the output directory from `../small/..` to
# `../big/..` so that it does not destroy the result you obtained in local mode.
# Run the .py file the cluster mode to calculate cluster centers and Mirai percentage for each cluster.
# - Submit the .py file 
# - Submit the the log file that contains the run time information for a successful execution in the cluster mode.
# - Submit the output file that records the cluster summary in the local mode.
# - Submit the output file that records the cluster summary in the cluster mode.

# In[93]:


ss.stop()


# In[ ]:




