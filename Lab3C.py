#!/usr/bin/env python
# coding: utf-8

# # DS/CMPSC 410 Fall 2022
# ## Instructor: Professor John Yen
# ## TAs: Rupesh Prajapati and Haiwen Guan
# ## LAs: Zining Yin
# 
# # Lab 3: Hashtag Counting and Spark-submit in Cluster Mode
# ## The goals of this lab are for you to be able to
# ## - Use the RDD transformations ``filter`` and ``sortBy``.
# ## - Compute hashtag counts for an input data file containing tweets.
# ## - Modify PySpark code for local mode into PySpark code for cluster mode.
# ## - Request a cluster from ICDS, and run spark-submit in the cluster mode for a big dataset.
# ## - Obtain run-time performance for a choice on the number of output partitions for reduceByKey.
# ## - Apply the obove to compute hashtag counts for tweets related to Boston Marathon Bombing (gathered on April 17, 2013, two days after the domestic terrorist attack).
# 
# ## Total Number of Exercises: 5
# - Exercise 1: 5 points
# - Exercise 2: 10 points
# - Exercise 3: 10 points
# - Exercise 4: 15 points
# - Exercise 5: 10 points
# 
# ## Total Points: 50 points
# 
# ## Data for Lab 3
# - sampled_4_17_tweets.csv : A random sampled of a small set of tweets regarding Boston Marathon Bombing on April 17, 2013. This data is used in the local mode.
# - BMB_4_17_tweets.csv : The entire set of tweets regarding Boston Marathon Bombing on April 17, 2013. This data is used in the cluster mode.
# - Like Lab2, download the data from Canvas into a directory for the lab (e.g., Lab3) under your home directory.
# 
# ## Items to submit for Lab 3
# - Completed Jupyter Notebook (HTML format)
# - .py file used for cluster mode
# - log file for a successful run in the cluster mode
# - a screen shot of the ``ls -al`` command in the output directory for a successful run in the cluster mode.
# 
# # Due: midnight, Sept 11, 2022

# ## Like Lab 2, the first thing we need to do in each Jupyter Notebook running pyspark is to import pyspark first.

# In[1]:


import pyspark


# In[2]:


from pyspark import SparkContext


# ## Like Lab 2, ww create a Spark Context object.  
# 
# - Note: We use "local" as the master parameter for ``SparkContext`` in this notebook so that we can run and debug it in ICDS Jupyter Server.  However, we need to remove ``"master="local",``later when you convert this notebook into a .py file for running it in the cluster mode.

# In[3]:


sc=SparkContext(master="local", appName="Lab3")
sc


# # Exercise 1 (5 points)  Add your name below 
# ## Answer for Exercise 1
# - Your Name:

# In[4]:


#Name: Kashish Gujral


# # Exercise 2 (10 points) 
# ## Complete the path and run the code below to read the file "sampled_4_17_tweets.csv" from your Lab3 directory.

# In[5]:


tweets_RDD = sc.textFile("/storage/home/kmg6272/Lab3/sampled_4_17_tweets.csv")
tweets_RDD


# # Exercise 3 (10 points) 
# ## Complete and execute the code below, which computes the total count of hashtags in the input tweets, sort them by count (in descending order), and save them in an output directory:
# - (a) Uses flatMap to "flatten" the list of tokens from each tweet (using split function) into a very large list of tokens.
# - (b) Filter the token for hashtags.
# - (c) Count the total number of hashtags in a way similar to Lab 2.
# - (d) Sort the hashtag count in descending order.
# - (e) Save the sorted hashtag counts in an output directory.

# ## Code for Exercise 3 is shown in the Code Cells below.

# In[6]:


tokens_RDD = tweets_RDD.flatMap(lambda line: line.strip().split(" "))
tokens_RDD.take(3)


# # take (action for RDD)
# - ``take`` is an action for RDD.  
# - The parameter is the number of elements from the input RDD you want to show.
# - `take` is often used for debugging/learning purpose in the local mode so that the contents of a few samples of an RDD can be revealed.  This way, if the content and/or the format of the RDD differs from what you expected, you can investigate it and, if needed, fix it before proceeding further.

# # filter (transformation for RDD)
# 
# - The syntax for filtering (one type of data trasnformation in Spark) an input RDD is
# ``<input RDD>.filter(lambda <parameter> : <the body of a Boolean function> )``
# - Notice the syntax is not what is described in p. 38 of the textbook.
# - The result of filtering the input RDD is the collection of all elements in the input RDD that pass the filter condition (i.e., returns True when the filtering Boolean function is applied to each element of the input RDD). 
# - For example, the filtering condition in the pyspark conde below checks whether each element of the input RDD (i.e., `tokens_RDD`) starts with the character "#", using Python `startswith()` method for string.

# In[7]:


hashtag_RDD = tokens_RDD.filter(lambda x: x.startswith("#"))


# In[8]:


hashtag_1_RDD = hashtag_RDD.map(lambda x: (x, 1))


# In[9]:


hashtag_count_RDD = hashtag_1_RDD.reduceByKey(lambda x, y: x+y, 5 )


# In[ ]:





# # sortBy (transformation for RDD)
# - To sort hashtag count so that those that occur more frequent appear first, we use ``sortBy(lambda pair: pair[1], ascending=False)``.
# - `sortBy` sort the input RDD based on the value of the lambda function, which returns the value of the input key-value pair.  
# - Note: The index of a list/tuple in Python starts with 0. Therefore `pair[0]` accesses the key of each key-value pair (in the input RDD), whereas `pair[1]` accesses the value of the key-value pair in the input RDD.
# - The default sorting order is ascending. To sort in descending order, we need to set the parameter `ascending` to `False`, which means frequent/top hashtags occured first in the output.

# In[10]:


sorted_hashtag_count_RDD = hashtag_count_RDD.sortBy(lambda pair: pair[1], ascending=False)


# ### Note: You need to complete the path with your output directory (e.g., Lab3 under your home directory). 
# ### Note: You also need to change the directory names (e.g., replace "_sampled" with "_cluster") before you convert this notebook into a .py file for submiting it to ICDS cluster.  

# In[11]:


output_path = "/storage/home/kmg6272/Lab3/sorted_hashtag_count_sampled.txt" 
sorted_hashtag_count_RDD.saveAsTextFile(output_path)


# In[12]:


sc.stop()


# # Exercise 4 (15 points)
# After running this Jupyter Notebook successfully in local mode, modify the notebook in the following ways
# - Remove `master="local",` in SparkContext statement so that the Spark code runs in cluster (Stand Alone cluster) mode.
# - Change the input file to "BMB_4_17_tweets.csv", which you have downloaded from Canvas and saved in your directory for Lab3.
# - Change the output directory/path name 
# - Comment out the `take` action for RDD, which is used for debugging in the local mode.
# - Save the modified notebook as Lab3C.ipynb (using `Save Notebook as` under `File` on the top menu (top left).
# - Export the modified notebook using `Export Notebook as` under `File` to export it as `Executable Script` (the exported Python file has .py extension, e.g. Lab3C.py).
# - Upload Lab3C.py to the Lab3 directory in ICDS Roar under your home directory.
# - Follow the direction of "Instructions for Running Spark in Cluster Mode" to run Lab3C.py in the cluster mode using spark-submit.
# 
# - Submit .py file and a screen shot of `ls -al` in your output directory (in a Terminal window).

# # Exercise 5 (10 points)
# Record the run time information below and submit the log file that contains the run time information.

# ## Answer to Exercise 5
# - Number of partitions used in ReduceByKey:
# - real time:
# - user time:
# - sys  time:

# In[13]:


sc=SparkContext(appName="Lab3")
#sc


# In[14]:


tweets_RDD = sc.textFile("/storage/home/kmg6272/Lab3/BMB_4_17_tweets.csv")
#tweets_RDD


# In[15]:


tokens_RDD = tweets_RDD.flatMap(lambda line: line.strip().split(" "))
#tokens_RDD.take(3)


# In[16]:


hashtag_RDD = tokens_RDD.filter(lambda x: x.startswith("#"))


# In[17]:


hashtag_1_RDD = hashtag_RDD.map(lambda x: (x, 1))


# In[18]:


hashtag_count_RDD = hashtag_1_RDD.reduceByKey(lambda x, y: x+y, 20 )


# In[19]:


sorted_hashtag_count_RDD = hashtag_count_RDD.sortBy(lambda pair: pair[1], ascending=False)


# In[20]:


output_path = "/storage/home/kmg6272/Lab3/sorted_hashtag_count_cluster.txt" 
sorted_hashtag_count_RDD.saveAsTextFile(output_path)


# In[ ]:




