# Databricks notebook source
# MAGIC %md
# MAGIC ## Step 1) Download raw data from the github repo and copy it to DBFS. 
# MAGIC - We will also rename the folders in a way to make it easy to define an external table on top of the raw data.

# COMMAND ----------

# MAGIC %sh
# MAGIC # Dont Change this
# MAGIC # A simple shell script to download raw data from the github repo and changing the folder structure to make it accessible by the external tables.
# MAGIC # Debu Sinha - Aug/30/2021
# MAGIC 
# MAGIC rm -rf Spam-Review-Filter-NLP
# MAGIC git clone https://github.com/debu-sinha/Spam-Review-Filter-NLP.git
# MAGIC cd Spam-Review-Filter-NLP/data
# MAGIC rm README.md
# MAGIC mkdir -p negative_polarity/source=mturk/label=fake negative_polarity/source=web/label=real positive_polarity/source=mturk/label=fake positive_polarity/source=tripadvisor/label=real
# MAGIC mv negative_polarity/deceptive_from_MTurk/fold*/*.txt negative_polarity/source\=mturk/label\=fake/
# MAGIC mv negative_polarity/truthful_from_Web/fold*/*.txt negative_polarity/source\=web/label\=real/
# MAGIC mv positive_polarity/deceptive_from_MTurk/fold*/*.txt positive_polarity/source\=mturk/label\=fake/
# MAGIC mv positive_polarity/truthful_from_TripAdvisor/fold*/*.txt positive_polarity/source\=tripadvisor/label\=real/
# MAGIC rm -rf negative_polarity/deceptive_from_MTurk negative_polarity/truthful_from_Web positive_polarity/deceptive_from_MTurk positive_polarity/truthful_from_TripAdvisor
# MAGIC mv negative_polarity reviewpolarity=negative
# MAGIC mv positive_polarity reviewpolarity=positive
# MAGIC cd ..
# MAGIC cd .. 
# MAGIC rm -rf /dbfs/tmp/spam_review
# MAGIC mkdir -p /dbfs/tmp/spam_review

# COMMAND ----------

dbutils.fs.rm("dbfs:/tmp/spam_review", recurse=True)
dbutils.fs.mv("file:/databricks/driver/Spam-Review-Filter-NLP/data", "dbfs:/tmp/spam_review/", recurse=True)
dbutils.fs.rm("dbfs:/tmp/spam_review/data", recurse=True)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 2) Define Database that will contain all the tables we will be working with in this project.
# MAGIC - We will also define an external table on top of the raw dataset to make it easy to work with.

# COMMAND ----------

# MAGIC %sql
# MAGIC DROP DATABASE spam_reviews CASCADE;
# MAGIC 
# MAGIC CREATE DATABASE IF NOT EXISTS spam_reviews;
# MAGIC USE spam_reviews;
# MAGIC 
# MAGIC CREATE TABLE IF NOT EXISTS spam_reviews.spam_reviews_raw(review string) 
# MAGIC USING com.databricks.spark.csv
# MAGIC COMMENT 'This table has raw data for the spamreview_demo' 
# MAGIC PARTITIONED  BY(reviewpolarity string, source string, label string)
# MAGIC LOCATION 'dbfs:/tmp/spam_review'
# MAGIC OPTIONS (delimiter "\n",header "false", inferSchema "false");
# MAGIC 
# MAGIC MSCK REPAIR TABLE spam_reviews.spam_reviews_raw;

# COMMAND ----------

# MAGIC %sql
# MAGIC select * from spam_reviews.spam_reviews_raw

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 3) Install the desired Spark NLP libraries on the current cluster. You can read more about different ways to install libraries on Databricks [here](https://docs.databricks.com/libraries/index.html). You can read more about John Snow Labs Spark NLP Databricks support [here](https://nlp.johnsnowlabs.com/docs/en/install#databricks-support)
# MAGIC - Spark NLP Maven: com.johnsnowlabs.nlp:spark-nlp_2.12:3.2.2
# MAGIC - Spark NLP PyPi: spark-nlp

# COMMAND ----------

# MAGIC %md
# MAGIC <b>Pro-tip:</b> For python notebook you can also also make use of notebook scoped libraries by using <b>%pip</b> construct in Databricks. More can be found [here](https://docs.databricks.com/libraries/notebooks-python-libraries.html#notebook-scoped-python-libraries)
