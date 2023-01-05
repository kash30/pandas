#!/usr/bin/env python
# coding: utf-8

# # DS/CMPSC 410 MiniProject Deliverable #3
# 
# # Fall 2022
# ### Instructor: Prof. John Yen
# ### TA: Rupesh Prajapati and Haiwen Guan
# ### LA: Zining Yin
# 
# ### Learning Objectives
# - Be able to apply PCA to reduce the high dimensional feature space to facilitate ML for high dimensional data.
# - Like Minproject Deliverable #2, focus on the clustering of non-extreme multi-port scanners based on the ports they scanned.
# - Be able to choose a suitable threshold for non-extreme scanners based on the results of miniproject deliverable #2.
# - Be able to choose a larger k value for finding top k ports, which are used for one-hot encoding.
# - Be able to perform k-means with and without PCA, and compare the results using silhouette score and mirai external labels.
# - Be able to obtain cluster centers of the original feature space for clustering results using PCA and k-means.
# - After successful clustering of the small Darknet dataset (with and without PCA), conduct clustering on the large Darknet dataset in the cluster mode.
# 
# ### Total points: 100 
# - Exercise 1: 5 points
# - Exercise 2: 5 points 
# - Exercise 3: 5 points 
# - Exercise 4: 10 points
# - Exercise 5: 10 points
# - Exercise 6: 10 points
# - Exercise 7: 15 points
# - Exercise 8: 40 points
#   
# ### Due: 11:59 pm, November 27, 2022
# ### Early Submission bonus (before midnight November 25th): 10 points

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
from pyspark.ml.feature import PCA


# In[3]:


ss = SparkSession.builder.master("local").appName("MiniProject 3 k-meas Clustering using OHE").getOrCreate()


# ## Exercise 1 (5 points)
# Complete the path for input file in the code below and enter your name in this Markdown cell:
# - Name: Kashish Gujral
# ### Note: You will need to change the name of the input file in the cluster mode to `Day_2020_profile.csv`

# In[4]:


Scanners_df = ss.read.csv("/storage/home/kmg6272/work/MiniProj3/sampled_profile.csv", header= True, inferSchema=True )


# ## We can use printSchema() to display the schema of the DataFrame Scanners_df to see whether it was inferred correctly.

# In[5]:


Scanners_df.printSchema()


# # In this lab, our goal is to answer the question:
# ## Q: What groups of scanners are similar in the ports they scan?
# 
# ### Because we know (from MiniProject 1) about two third of the scanners scan only 1 port, we can separate them from the other scanners.

# ### Because the feature `numports` record the total number of ports being scanned by each scanner, we can use it to separate 1-port-scanners from multi-port-scanners.

# In[6]:


multi_port_scanners= Scanners_df.where(col("numports")>1)


# In[7]:


multi_port_scanners_count = multi_port_scanners.count()


# In[8]:


print(multi_port_scanners_count)


# In[9]:


ScannersCount_byNumPorts = multi_port_scanners.groupby("numports").count()


# In[10]:


ScannersCount_byNumPorts.show(3)


# In[11]:


SortedNumPorts_DF= ScannersCount_byNumPorts.orderBy("numports", ascending = False)


# In[12]:


SortedNumPorts_DF.show(70)


# In[13]:


max_non_rare_NumPorts= 1000


# ## We are going to focus on the grouping of scanners that scan at least two ports, and do not scan very large number of ports. We will call these scanners Non-extreme Multi-port Scanners.
# ## We will save the extreme scanners in a csv file so that we can process it separately.

# In[14]:


extreme_scanners = Scanners_df.where(col("numports") > max_non_rare_NumPorts)


# In[15]:


path2="/storage/home/kmg6272/work/MiniProj3/small/Extreme_Scanners.csv"
extreme_scanners.write.option("header",True).csv(path2)


# In[16]:


non_extreme_multi_port_scanners = Scanners_df.where(col("numports") <= max_non_rare_NumPorts).where(col("numports") > 1)


# In[17]:


non_extreme_multi_port_scanners.count()


# # Part A: One Hot Encoding of Top 200 Ports
# We want to apply one hot encoding to the top 200 ports scanned by scanners. 
# - This is similar to MiniProject 2, with one difference (we use top 200 ports in MiniProject 3, rather than top 100 ports).

# In[18]:


non_extreme_multi_port_scanners.select("ports_scanned_str").show(4)


# # For each port scanned, count the Total Number of Scanners that Scan the Given Port
# Like MiniProject 1, to calculate this, we need to 
# - (a) convert the ports_scanned_str into an array/list of ports
# - (b) Convert the DataFrame into an RDD
# - (c) Use flatMap to count the total number of scanners for each port.

# # The Following Code Implements the three steps.
# ## (a) Split the column "Ports_Array" into an Array of ports.

# In[19]:


# (a)
NEMP_Scanners_df=non_extreme_multi_port_scanners.withColumn("Ports_Array", split(col("ports_scanned_str"), "-") )
NEMP_Scanners_df.show(10)


# ## (b) We convert the column ```Ports_Array``` into an RDD so that we can apply flatMap for counting.

# In[20]:


Ports_Scanned_RDD = NEMP_Scanners_df.select("Ports_Array").rdd


# In[21]:


Ports_Scanned_RDD.take(5)


# ## (c) Because each port number in the Ports_Array column for each row/scanner occurs only once, we can count the total number of scanners by counting the total occurance of each port number through flatMap.
# ### Because each element of the rdd is a Row object, we need to first extract the first element of the row object, which is the list of Ports, from the ``Ports_Scanned_RDD``
# ### We can then count the total number of occurance of a port using map and reduceByKey, like counting word/hashtag frequency in tweets.

# In[22]:


Ports_Scanned_RDD.take(3)


# In[23]:


Ports_list_RDD = Ports_Scanned_RDD.map(lambda row: row[0] )


# In[24]:


Ports_list_RDD.take(3)


# In[25]:


Ports_list2_RDD = Ports_list_RDD.flatMap(lambda x: x )


# In[26]:


Ports_list2_RDD.take(7)


# In[27]:


Port_count_RDD = Ports_list2_RDD.map(lambda x: (x, 1))
Port_count_RDD.take(7)


# In[28]:


Port_count_total_RDD = Port_count_RDD.reduceByKey(lambda x,y: x+y)
Port_count_total_RDD.take(5)


# In[29]:


Port_count_total_RDD.count()


# ## Note: These are the total number of ports scanned by Non-extreme Multi-port scanners in the small dataset.  Is the result what you expected?

# ## Exercise 2 (5 points) Complete the code below to finds top k ports scanned by non-extreme multi-port scanners. We set k to 200 for Mini-project 3.

# In[30]:


Sorted_Count_Port_RDD = Port_count_total_RDD.map(lambda x: (x[1], x[0])).sortByKey( ascending = False)


# In[31]:


top_k_ports = 200
Sorted_Count_Port_RDD.take(top_k_ports)


# In[32]:


Sorted_Ports_RDD= Sorted_Count_Port_RDD.map(lambda x: x[1] )
Top_Ports_list = Sorted_Ports_RDD.take(top_k_ports)


# In[33]:


Top_Ports_list


# #  A.2 One Hot Encoding of Top K Ports
# ## One-Hot-Encoded Feature/Column Name
# Because we need to create a name for each one-hot-encoded feature, which is one of the top k ports, we can adopt the convention that the column name is "PortXXXX", where "XXXX" is a port number. This can be done by concatenating two strings using ``+``.

# In[34]:


Top_Ports_list[199]


# In[35]:


FeatureName = "Port"+Top_Ports_list[199]


# In[36]:


FeatureName


# ## One-Hot-Encoding using withColumn and array_contains

# In[37]:


from pyspark.sql.functions import array_contains


# ## Generate Hot-One Encoded Feature for each of the top k ports in the Top_Ports_list
# 
# - Iterate through the Top_Ports_list so that each top port is one-hot encoded into the DataFrame for non-extreme multi-port scanners (i.e., `NEMP_Scanners2.df`).

# ## Exercise 3 (5 points) Complete the following PySpark code for encoding the n ports using One Hot Encoding, where n is specified by the variable ```top_k_ports```

# In[38]:


top_k_ports


# In[39]:


Top_Ports_list[top_k_ports - 1]


# In[40]:


# Initialize NEMP_Scanners2_df
NEMP_Scanners2_df = NEMP_Scanners_df
for i in range(0, top_k_ports):
    # "Port" + Top_Ports_list[i]  is the name of each new feature created through One Hot Encoding Top_Ports_list
    NEMP_Scanners3_df = NEMP_Scanners2_df.withColumn("Port" + Top_Ports_list[i], array_contains("Ports_Array",Top_Ports_list[i]))
    NEMP_Scanners2_df = NEMP_Scanners3_df


# In[41]:


NEMP_Scanners2_df.printSchema()


# ## Exercise 4 (10 points)  Complete the code below to use k-means to cluster non-extreme multi-port scanners using one-hot-encoded top 200 ports.

# # Specify One-Hot Encoded Top k Ports as Input Features for k Means Clustering

# In[42]:


input_features = [ ]
for i in range(0, top_k_ports ):
    input_features.append( "Port"+ Top_Ports_list[i])


# In[43]:


print(input_features)


# # Part B k-Means Clustering (k=200)

# In[44]:


va = VectorAssembler().setInputCols(input_features).setOutputCol("features")


# In[45]:


data= va.transform(NEMP_Scanners2_df)


# In[46]:


data.show(3)


# In[48]:


data.persist()


# In[49]:


total_clusters = 200
km = KMeans(featuresCol= "features", predictionCol="prediction").setK(200).setSeed(123)
km.explainParams()


# In[50]:


kmModel=km.fit(data)


# In[51]:


kmModel


# In[52]:


predictions = kmModel.transform(data)


# In[53]:


predictions.show(3)


# In[54]:


Cluster1_df=predictions.where(col("prediction")==0)


# In[55]:


Cluster1_df.count()


# In[56]:


summary = kmModel.summary


# In[57]:


summary.clusterSizes


# In[58]:


evaluator = ClusteringEvaluator()
silhouette = evaluator.evaluate(predictions)


# In[59]:


print('Silhouette Score of the Clustering Result is ', silhouette)


# In[60]:


centers = kmModel.clusterCenters()


# In[61]:


print(centers)


# # Record cluster index, cluster size, percentage of Mirai scanners, and cluster centers for each clusters formed.
# ## The value of cluster center for a OHE top port is the percentage of data/clusters in the cluster that scans the top port. For example, a cluster center `[0.094, 0.8, 0, ...]` indicates the following
# - 9.4% of the scanners in the cluster scan Top_Ports_list[0]: port 17132
# - 80% of the scanners in the cluster scan Top_Ports_list[1]: port 17130
# - No scanners in the cluster scan Top_Ports_list[2]: port 17140

# # Exercise 5 (10 points) Complete the code below for computing the percentage of Mirai scanners for each scanner, and record it together with cluster centers for each cluster (without PCA).

# In[62]:


import pandas as pd
import numpy as np
import math


# In[63]:


# Define columns of the Pandas dataframe
column_list = ['cluster ID', 'size', 'mirai_ratio' ]
for feature in input_features:
    column_list.append(feature)
clusters_summary_df = pd.DataFrame( columns = column_list )
for i in range(0, total_clusters):
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


# In[64]:


path4= "/storage/home/kmg6272/work/MiniProj3/small/Clusters_Summary_200OHE_k200.csv"
clusters_summary_df.to_csv(path4, header=True)


# # Part C Using PCA for Dimension Reduction

# # We use PCA to reduce the input dimension from 200 to 100.

# In[65]:


pca_model = PCA(k=100, inputCol = "features", outputCol="pca_features")


# # The `pca_model` is a template for constructing a PCA model.
# ## After we apply `fit` to a PCA template, we obtain an actual mapping from the original feature space to the PCA space.

# In[66]:


p_model = pca_model.fit(data)


# In[67]:


p_data = p_model.transform(data)


# In[68]:


p_data.persist()


# ## Notice we change the `featuresCol` for k-means clustering to `"pca_features"` because we want to use its reduced dimension for clustering.

# In[69]:


total_clusters = 200
km2 = KMeans(featuresCol= "pca_features", predictionCol="pca_prediction").setK(total_clusters).setSeed(123)
km2.explainParams()


# In[70]:


kmModel_p=km2.fit(p_data)


# In[71]:


kmModel_p


# In[72]:


p_predictions = kmModel_p.transform(p_data)


# In[73]:


p_predictions.persist()


# In[74]:


summary_p = kmModel_p.summary


# In[75]:


summary_p.clusterSizes


# # Notice that we need to specify `featuresCol` explicitly because the default is "features", but we are using "pca_features" as input features for Part C.

# In[76]:


evaluator = ClusteringEvaluator(predictionCol="pca_prediction", featuresCol="pca_features")
silhouette = evaluator.evaluate(p_predictions)


# In[77]:


print('Silhouette Score of the Clustering Result is ', silhouette)


# In[78]:


top_k_ports


# In[79]:


import numpy as np
# Initialize the array to zeros
cluster_center_array = np.zeros([200,200])


# In[80]:


i=0
feature_name= "Port" + Top_Ports_list[i]
feature_i_count_by_clusters = p_predictions.where(col(feature_name)).groupBy("pca_prediction").count()


# In[81]:


fc_bc_rdd = feature_i_count_by_clusters.rdd


# In[82]:


fc_bc_list =fc_bc_rdd.collect()


# In[83]:


print(fc_bc_list)


# In[84]:


for row in fc_bc_list:
    print(row[0], row[1])


# In[86]:


p_predictions.persist()
for i in range(0, top_k_ports):
    feature_i_name = "Port" + Top_Ports_list[i]
    feature_i_count_by_clusters_DF = p_predictions.where(col(feature_i_name)).groupBy("pca_prediction").count()
    fic_bc_list = feature_i_count_by_clusters_DF.rdd.collect()
    print("Count of feature ", i, ": ", feature_i_name, "by clusters")
    print(" Cluster : ", fic_bc_list[0][0], "Count: ", fic_bc_list[0][1])
    for row in fic_bc_list:
        cluster_center_array[row[0]][i] = row[1]


# In[87]:


total_clusters = 200


# In[88]:


import pandas as pd


# In[89]:


# The number of total clusters (`total_clusters`) was specified earlier when we created k-means model template. 
# Define columns of the Pandas dataframe
column_list = ['cluster ID', 'size', 'mirai_ratio' ]
for feature in input_features:
    column_list.append(feature)
clusters_summary_df = pd.DataFrame( columns = column_list )
for i in range(0, total_clusters):
    cluster_i = p_predictions.where(col('pca_prediction')== i)
    cluster_i.persist()
    cluster_i_size = cluster_i.count()
    cluster_i_mirai_count = cluster_i.where(col('mirai')).count()
    cluster_i_mirai_ratio = cluster_i_mirai_count/cluster_i_size
    if cluster_i_mirai_count > 0:
        print("Cluster ", i, "; Mirai Ratio:", cluster_i_mirai_ratio, "; Cluster Size: ", cluster_i_size)
    cluster_row = [i, cluster_i_size, cluster_i_mirai_ratio]
    for j in range(0, len(input_features)):
        # compute the center for the original jth feature (i.e., jth top port)
        feature_j = "Port" + Top_Ports_list[j]
        count_j = cluster_center_array[i][j]
        center_j = count_j / cluster_i_size
        cluster_row.append(center_j)
    clusters_summary_df.loc[i]= cluster_row
    cluster_i.unpersist()


# In[90]:


path5= "/storage/home/kmg6272/work/MiniProj3/small/Clusters_Summary_200OHE_PCA100_k200.csv"
clusters_summary_df.to_csv(path5, header=True)


# In[91]:


clusters_summary_df


# # Exercise 8 (40 points)
# Modify the Jupyter Notebook for running in cluster mode using the big dataset (Day_2020_profile.csv). Make sure you change the output directory from `../small/..` to
# `../big/..` so that it does not destroy the result you obtained in local mode.
# Run the .py file the cluster mode to calculate cluster centers and Mirai percentage for each cluster with and without PCA.
# - Submit the .py file  (5 points)
# - Submit the the log file that contains the run time information for a successful execution in the cluster mode. (5 points)
# - Submit the output file that records the cluster summary in the cluster mode (without PCA) (10 points)
# - Submit the output file that records the cluster summary in the cluster mode (with PCA) (10 points)
# - Discuss the clustering results of k-means with PCA and without PCA (in a separate word document) (10 points)

# In[ ]:


ss.stop()


# In[ ]:




