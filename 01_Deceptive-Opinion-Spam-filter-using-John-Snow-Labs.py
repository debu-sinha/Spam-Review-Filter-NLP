# Databricks notebook source
# MAGIC %md 
# MAGIC # Detecting Spam reviews at scale using Apache Spark and John Snow Labs NLP on Databricks.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Required Databricks Runtime
# MAGIC - 8.4 ML (includes Apache Spark 3.1.2, Scala 2.12)
# MAGIC 
# MAGIC ## Required Libraries
# MAGIC - Java Libraries
# MAGIC    - Spark NLP -> com.johnsnowlabs.nlp:spark-nlp_2.12:3.2.2
# MAGIC - Python Libraries
# MAGIC    - Spark NLP -> spark-nlp==3.2.2 
# MAGIC 
# MAGIC For questions reach out to debu.sinha@databricks.com | debusinha2009@gmail.com

# COMMAND ----------

# MAGIC %md
# MAGIC ## About This Notebook
# MAGIC 
# MAGIC This notebooks is intended to help you understand how to architect and implement a NLP capable Dataprocessing pipeline at scale using Databricks and John Snow Labs. We will also train a Logistic regression model and track model training using MLFlow.  

# COMMAND ----------

# MAGIC %md 
# MAGIC ## Dataset
# MAGIC Deceptive Opinion Spam Corpus v1.4
# MAGIC 
# MAGIC ### Overview
# MAGIC --------
# MAGIC 
# MAGIC This corpus consists of truthful and deceptive hotel reviews of 20 Chicago
# MAGIC hotels. The data is described in two papers according to the sentiment of the
# MAGIC review. In particular, we discuss positive sentiment reviews in [1] and negative
# MAGIC sentiment reviews in [2].
# MAGIC 
# MAGIC While we have tried to maintain consistent data preprocessing procedures across
# MAGIC the data, there *are* differences which are explained in more detail in the
# MAGIC associated papers. Please see those papers for specific details.  
# MAGIC 
# MAGIC This corpus contains:
# MAGIC 
# MAGIC * 400 truthful positive reviews from TripAdvisor (described in [1])
# MAGIC * 400 deceptive positive reviews from Mechanical Turk (described in [1])
# MAGIC * 400 truthful negative reviews from Expedia, Hotels.com, Orbitz, Priceline,
# MAGIC   TripAdvisor and Yelp (described in [2])
# MAGIC * 400 deceptive negative reviews from Mechanical Turk (described in [2])
# MAGIC 
# MAGIC Each of the above datasets consist of 20 reviews for each of the 20 most popular
# MAGIC Chicago hotels (see [1] for more details). The files are named according to the
# MAGIC following conventions:
# MAGIC 
# MAGIC * Directories prefixed with `fold` correspond to a single fold from the
# MAGIC   cross-validation experiments reported in [1] and [2].		
# MAGIC 
# MAGIC * Files are named according to the format `%c_%h_%i.txt`, where:
# MAGIC 
# MAGIC     * %c denotes the class: (t)ruthful or (d)eceptive
# MAGIC 
# MAGIC     * %h denotes the hotel:
# MAGIC 
# MAGIC         * affinia: Affinia Chicago (now MileNorth, A Chicago Hotel)
# MAGIC         * allegro: Hotel Allegro Chicago - a Kimpton Hotel
# MAGIC         * amalfi: Amalfi Hotel Chicago
# MAGIC         * ambassador: Ambassador East Hotel (now PUBLIC Chicago)
# MAGIC         * conrad: Conrad Chicago
# MAGIC         * fairmont: Fairmont Chicago Millennium Park
# MAGIC         * hardrock: Hard Rock Hotel Chicago
# MAGIC         * hilton: Hilton Chicago
# MAGIC         * homewood: Homewood Suites by Hilton Chicago Downtown
# MAGIC         * hyatt: Hyatt Regency Chicago
# MAGIC         * intercontinental: InterContinental Chicago
# MAGIC         * james: James Chicago
# MAGIC         * knickerbocker: Millennium Knickerbocker Hotel Chicago
# MAGIC         * monaco: Hotel Monaco Chicago - a Kimpton Hotel
# MAGIC         * omni: Omni Chicago Hotel
# MAGIC         * palmer: The Palmer House Hilton
# MAGIC         * sheraton: Sheraton Chicago Hotel and Towers
# MAGIC         * sofitel: Sofitel Chicago Water Tower
# MAGIC         * swissotel: Swissotel Chicago
# MAGIC         * talbott: The Talbott Hotel
# MAGIC 
# MAGIC     * %i serves as a counter to make the filename unique
# MAGIC 
# MAGIC Questions
# MAGIC ---------
# MAGIC 
# MAGIC Please direct questions to Myle Ott (<myleott@cs.cornell.edu>)
# MAGIC 
# MAGIC References
# MAGIC ----------
# MAGIC 
# MAGIC [1] M. Ott, Y. Choi, C. Cardie, and J.T. Hancock. 2011. Finding Deceptive
# MAGIC Opinion Spam by Any Stretch of the Imagination. In Proceedings of the 49th
# MAGIC Annual Meeting of the Association for Computational Linguistics: Human Language
# MAGIC Technologies.
# MAGIC 
# MAGIC [2] M. Ott, C. Cardie, and J.T. Hancock. 2013. Negative Deceptive Opinion Spam.
# MAGIC In Proceedings of the 2013 Conference of the North American Chapter of the
# MAGIC Association for Computational Linguistics: Human Language Technologies.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Architecture Diagram

# COMMAND ----------

# MAGIC %md
# MAGIC ![Architecture Diagram](https://raw.githubusercontent.com/debu-sinha/Spam-Review-Filter-NLP/dev/architecture.JPG)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1) Data Engineering

# COMMAND ----------

# MAGIC %md
# MAGIC ### 1.1) Configure the Environment
# MAGIC * Download dataset from the github repo and make it accessible as an Spark external table.
# MAGIC * Install `spark-nlp==3.2.2` as notebook scoped library using `%pip`
# MAGIC * Install  `com.johnsnowlabs.nlp:spark-nlp_2.12:3.2.2` as cluster scoped library fropm maven as described [here](https://docs.databricks.com/libraries/cluster-libraries.html#cluster-installed-library)
# MAGIC * set up database named `spam_reviews` and external table named `spam_reviews_raw`

# COMMAND ----------

# MAGIC %md
# MAGIC #####Note: Run cmd 10 only once.

# COMMAND ----------

# MAGIC %run ./00_setup

# COMMAND ----------

# MAGIC %pip install spark-nlp==3.2.2 mlflow

# COMMAND ----------

# MAGIC %md
# MAGIC <b>Pro-Tip:</b> In production environment use Databricks [Auto Loader](https://docs.databricks.com/spark/latest/structured-streaming/auto-loader.html).
# MAGIC 
# MAGIC But, what is Auto Loader?
# MAGIC 
# MAGIC - Auto Loader incrementally and efficiently loads new data files as they arrive in S3 or Azure Blog Storage. This is enabled by providing a Structured Streaming source called cloudFiles.
# MAGIC 
# MAGIC - Auto Loader internally keeps tracks of what files have been processed to provide exactly-once semantics, so you do not need to manage any state information yourself.
# MAGIC 
# MAGIC - Auto Loader supports two modes for detecting when new files arrive:
# MAGIC 
# MAGIC   - Directory listing: Identifies new files by parallel listing of the input directory. Quick to get started since no permission configurations are required. Suitable for scenarios where only a few files need to be streamed in on a regular basis.
# MAGIC 
# MAGIC   - File Notification: Uses AWS SNS and SQS services that subscribe to file events from the input directory. Auto Loader automatically sets up the AWS SNS and SQS services. File notification mode is more performant and scalable for large input directories.
# MAGIC 
# MAGIC - In this notebook, we use Directory Listing as that is the default mode for detecting when new files arrive.

# COMMAND ----------

# MAGIC %md
# MAGIC ### 1.2) Read raw data into a Delta table and create Bronze layer table.

# COMMAND ----------

# MAGIC %md #### Getting started with <img src="https://docs.delta.io/latest/_static/delta-lake-logo.png" width=300/>
# MAGIC 
# MAGIC An open-source storage layer for data lakes that brings ACID transactions to Apache Spark™ and big data workloads.
# MAGIC 
# MAGIC * ***ACID Transactions***: Ensures data integrity and read consistency with complex, concurrent data pipelines.
# MAGIC * ***Unified Batch and Streaming Source and Sink***: A table in Delta Lake is both a batch table, as well as a streaming source and sink. Streaming data ingest, batch historic backfill, and interactive queries all just work out of the box. 
# MAGIC * ***Schema Enforcement and Evolution***: Ensures data cleanliness by blocking writes with unexpected.
# MAGIC * ***Time Travel***: Query previous versions of the table by time or version number.
# MAGIC * ***Deletes and upserts***: Supports deleting and upserting into tables with programmatic APIs.
# MAGIC * ***Open Format***: Stored as Parquet format in blob storage.
# MAGIC * ***Audit History***: History of all the operations that happened in the table.
# MAGIC * ***Scalable Metadata management***: Able to handle millions of files are scaling the metadata operations with Spark.

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE TABLE IF NOT EXISTS spam_reviews.spam_reviews_delta_bronze
# MAGIC USING DELTA
# MAGIC AS SELECT row_number() OVER(ORDER BY reviewpolarity, label, source) AS sno, review, reviewpolarity, label, source FROM spam_reviews.spam_reviews_raw;

# COMMAND ----------

# MAGIC %md
# MAGIC ### 1.3) Read data from `spam_reviews.spam_reviews_delta_bronze` as a Spark Dataframe.

# COMMAND ----------

data =  spark.sql("select sno, review, label from spam_reviews.spam_reviews_delta_bronze order by sno")
display(data)

# COMMAND ----------

#total number of records
data.count()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2) Feature Engineering

# COMMAND ----------

# MAGIC %md 
# MAGIC ### 2.1) Import libraries

# COMMAND ----------

from  pyspark.ml.feature import Normalizer as VectorNormalizer
from pyspark.ml.classification import LogisticRegression
from pyspark.ml import *
from pyspark.ml.feature import *
from pyspark.sql.functions import *
from sparknlp.annotator import *
from sparknlp.common import *
from sparknlp.base import *
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
import sparknlp

# COMMAND ----------

#Divide the dataset into train and test
train_df, test_df = data.randomSplit([.8, .2], seed=42)
train_x = train_df.sort(train_df.sno).select("sno", "review")
train_y = train_df.sort(train_df.sno).select("sno", "label")
test_x = test_df.sort(test_df.sno).select("sno", "review")
test_y = test_df.sort(test_df.sno).select("sno", "label")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2.2 One-hot encode feature categorical columns.
# MAGIC 
# MAGIC We will convert each catgorical independent feature into multiple binary columns.
# MAGIC For each input string column, the number of output columns is equal to the number of unique values in the input column.
# MAGIC This is used for string columns with relatively few unique values.

# COMMAND ----------

#dealing with the training independent features
string_col_list = [i[0] for i in train_x.dtypes if ((i[1] == 'string') and (i[0] != "review"))]
#convert string to integer
indexers  = [StringIndexer(inputCol = categorical_col, outputCol = categorical_col + "_index") for categorical_col in string_col_list]
#one hot encoding all the categorical columns except the label column
encoders = [OneHotEncoder(inputCols=[categorical_col+"_index"], outputCols=[categorical_col+"_vec"]) for categorical_col in string_col_list if categorical_col != "label"]

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2.3 Process the `review` column

# COMMAND ----------

# To get through the process in Spark NLP, we need to get raw data transformed into Document type first.
# DocumentAssembler() is a special transformer that does this for us 
# it creates the first annotation of type Document which may be used by annotators down the road.
document_assembler = DocumentAssembler().setInputCol("review").setOutputCol("document")

# Tokenization is the process of splitting a text into smaller units(tokens). 
tokenizer = Tokenizer().setInputCols(["document"]).setOutputCol("tokens") 

# Remove Stop words. a, an, the, for, where etc 
stop_words_cleaner = StopWordsCleaner.pretrained().setInputCols("tokens").setOutputCol("cleanTokens").setCaseSensitive(False)

# remove punctuations (keep alphanumeric chars)
# if we don't set CleanupPatterns, it will only keep alphabet letters ([^A-Za-z])
normalizer = Normalizer().setInputCols(["cleanTokens"]).setOutputCol("normalized").setLowercase(True).setCleanupPatterns(["""[^\w\d\s]"""]) 

#reduce words to morpheme or base form. This is useful to reduce data dimensionality and data cleaning.
#Lemmatizer reduces a token to its lemma. Example : running, ran, run, rans will be reduced to run.
lemmatizer = LemmatizerModel.pretrained().setInputCols(["normalized"]).setOutputCol("lemmatized") 

# Converts annotation results into a format that easier to use.
#It is useful to extract the results from Spark NLP Pipelines. The Finisher outputs annotation(s) values into array.
finisher = Finisher().setInputCols(["lemmatized"]).setOutputAsArray(True)

#Count vectorizer will create a document term matrix, A matrix that lists out out of all the unique words in our corpus and then calculates 
#how many times a word appears in a document. 
# Extracts a vocabulary from document collections and generates a CountVectorizerModel. https://spark.apache.org/docs/latest/api/python/reference/api/pyspark.ml.feature.CountVectorizer.html
cv = CountVectorizer(inputCol='finished_lemmatized', outputCol="features", vocabSize=10000)

#Compute the Inverse Document Frequency (IDF) given a collection of documents. https://spark.apache.org/docs/latest/api/python/reference/api/pyspark.ml.feature.IDF.html?highlight=idf#pyspark.ml.feature.IDF.minDocFreq
idf = IDF(inputCol="features", outputCol="sparse_vecs_notnorm")

#different from Normalizer from Spark NLP. This Normalizer normalizes a vector to have unit norm using the given p-norm.https://spark.apache.org/docs/latest/api/python/reference/api/pyspark.ml.feature.Normalizer.html?highlight=normalizer#pyspark.ml.feature.Normalizer.p. p=2 is l2 norm.
vector_normalizer =  VectorNormalizer(inputCol="sparse_vecs_notnorm", outputCol="sparse_vecs", p=2)


#tying it all into feature engineering pipeline
feature_engineering_pipeline = Pipeline(stages=indexers + encoders+ [document_assembler, tokenizer, stop_words_cleaner, normalizer, lemmatizer, finisher, cv, idf, vector_normalizer])

# COMMAND ----------

# we will use this in future when creating our final model pipeline.
feature_engineering_tranformer = feature_engineering_pipeline.fit(train_x)


# COMMAND ----------

# MAGIC %md
# MAGIC ### 2.4 Process the `label`column

# COMMAND ----------

label_indexer_estimator = StringIndexer(inputCol = "label", outputCol = "label" + "_index")
label_indexer_transformer = label_indexer_estimator.fit(train_y)
binarized_train_y = label_indexer_transformer.transform(train_y)
binarized_test_y = label_indexer_transformer.transform(test_y)

# COMMAND ----------

combined_feature_engineered_training_df =  train_x.join(binarized_train_y, train_x.sno == binarized_train_y.sno,"inner" ).select("review", "label_index")
combined_feature_engineered_test_df =  test_x.join(binarized_test_y, test_x.sno == binarized_test_y.sno,"inner" ).select("review", "label_index")

# COMMAND ----------

# MAGIC %md 
# MAGIC ## 3) Train classification model
# MAGIC - Create an MLFlow experiment to track the runs and set as current
# MAGIC - Change the model parameters and re-run the training cell to log a different trial to the MLflow experiment
# MAGIC - To view the full list of tunable hyperparameters, check the output of the cell below

# COMMAND ----------

# MAGIC %md 
# MAGIC ### 3.1) Create MLFlow experiment

# COMMAND ----------

import mlflow

user = dbutils.notebook.entry_point.getDbutils().notebook().getContext().tags().apply('user') # user name associated with your account. its usually an email
username = user.split("@")[0]
mlflow_experiment_name = '{}_spam_review_classifier'.format(username)
experiment_location = '/Users/{}/{}'.format(user, mlflow_experiment_name) #experiment will be listed in this directory.
try:
  mlflow.create_experiment(name=experiment_location)
except:
  #experiment already exist
  # to delete the existing experiment uncomment the next lines. Read more:   https://mlflow.org/docs/latest/python_api/mlflow.html#mlflow.get_experiment_by_name
  mlflow.delete_experiment(mlflow.get_experiment_by_name(name=experiment_location).experiment_id)
  mlflow.create_experiment(name=experiment_location)
  mlflow.set_experiment(experiment_name=experiment_location)

dbutils.fs.rm("dbfs:/tmp/"+mlflow_experiment_name, recurse=True)  
  

# COMMAND ----------

from pyspark.ml.classification import LogisticRegression

help(LogisticRegression)

# COMMAND ----------

from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

lr = LogisticRegression(featuresCol="sparse_vecs", labelCol='label_index', maxIter=30)
#final estimator 
model = Pipeline(stages=feature_engineering_pipeline.getStages() + [lr])

# COMMAND ----------

evaluator = MulticlassClassificationEvaluator(predictionCol='prediction', labelCol='label_index', metricName='f1')
mlflow.spark.autolog()

with mlflow.start_run(run_name="logistic_regression") as mlflow_run:    
    model_transformer = model.fit(combined_feature_engineered_training_df)
    train_prediction = model_transformer.transform(combined_feature_engineered_training_df)
    test_prediction = model_transformer.transform(combined_feature_engineered_test_df)
    
    mlflow.log_metric("training_f1", evaluator.evaluate(train_prediction))
    mlflow.log_metric("test_f1", evaluator.evaluate(test_prediction))
    
    mlflow.spark.log_model(spark_model=model_transformer, artifact_path="dbfs:/tmp/"+mlflow_experiment_name)
     

# COMMAND ----------

# MAGIC %md
# MAGIC <b>Pro-tip:</b> Databricks also ships with [Hyperopt](https://docs.databricks.com/applications/machine-learning/automl-hyperparam-tuning/index.html) to parallelize model search using hyper parameter tuning.

# COMMAND ----------


