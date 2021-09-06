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

# MAGIC %pip install spark-nlp==3.2.2

# COMMAND ----------

# MAGIC %run ./00_setup

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

data =  spark.sql("select sno, review, reviewpolarity from spam_reviews.spam_reviews_delta_bronze")
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

from pyspark.ml import Pipeline
from pyspark.ml.feature import *
from pyspark.sql.functions import *
from sparknlp.annotator import *
from sparknlp.common import *
from sparknlp.base import *
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.classification import LogisticRegression
import sparknlp

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2.2 Convert the categorical columns to numeric excluding the `review` column and One-hot encoding non target column.
# MAGIC 
# MAGIC We will convert each string column into multiple binary columns.
# MAGIC For each input string column, the number of output columns is equal to the number of unique values in the input column.
# MAGIC This is used for string columns with relatively few unique values.

# COMMAND ----------

string_col_list = [i[0] for i in data.dtypes if ((i[1] == 'string') and (i[0] != "review"))]
#convert string to integer
string_indexers = [StringIndexer(inputCol = categorical_col, outputCol = categorical_col + "_index") for categorical_col in string_col_list]
#one hot encoding all the categorical columns except the label column
encoders = [OneHotEncoder(inputCols=[categorical_col+"_index"], outputCols=[categorical_col+"_vec"]) for categorical_col in string_col_list if categorical_col != "label"]

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2.3 Process the review column

# COMMAND ----------

# To get through the process in Spark NLP, we need to get raw data transformed into Document type first.
# DocumentAssembler() is a special transformer that does this for us 
# it creates the first annotation of type Document which may be used by annotators down the road.
document_assembler = DocumentAssembler().setInputCol("review").setOutputCol("document")

# Tokenization is the process of splitting a text into smaller units(tokens). 
tokenizer = Tokenizer().setInputCols(["document"]).setOutputCol("token") 

# Remove Stop words. a, an, the, for, where etc 
stop_words_cleaner = StopWordsCleaner.pretrained().setInputCols("token").setOutputCol("cleanTokens").setCaseSensitive(False)

# remove punctuations (keep alphanumeric chars)
# if we don't set CleanupPatterns, it will only keep alphabet letters ([^A-Za-z])
normalizer = Normalizer().setInputCols(["cleanTokens"]).setOutputCol("normalized").setLowercase(True).setCleanupPatterns(["""[^\w\d\s]"""]) 

#reduce words to morpheme or base form. This is useful to reduce data dimensionality and data cleaning.
#Lemmatizer reduces a token to its lemma. Example : running, ran, run, rans will be reduced to run.
lemmatizer = LemmatizerModel.pretrained().setInputCols(["normalized"]).setOutputCol("lemmatized") 

# Converts annotation results into a format that easier to use.
#It is useful to extract the results from Spark NLP Pipelines. The Finisher outputs annotation(s) values into array.
finisher = Finisher().setInputCols(["lemmatized"]).setOutputAsArray(True)

#tying it all into feature engineering pipeline
feature_engineering_pipeline = Pipeline(stages=[string_indexers, encoders, document_assembler, token_assembler, normalizer, lemmatizer_model, finisher])
feature_engineering_pipeline

# COMMAND ----------

feature_engineering_tranformer = feature_engineering_pipeline.fit(data)
feature_engineered_df = feature_engineering_tranformer.transform(data)

# COMMAND ----------

# MAGIC %md 
# MAGIC # 3) Trainings our machine learning model

# COMMAND ----------

from pyspark.ml.feature import *

# COMMAND ----------

cv = CountVectorizer(inputCol='finished_lemmatized', outputCol="features", vocabSize=8000)
idf = IDF(inputCol="features", outputCol="sparse_vecs_notnorm")
normalize = Normalizer(inputCol="sparse_vecs_notnorm", outputCol="sparse_vecs", p=2)
lr = LogisticRegression(featuresCol="sparse_vecs", labelCol='label_index', maxIter=30)

pipelineML = Pipeline(stages=[cv, idf, normalize, lr])
finalDf = pipelineML.fit(nlpDF).transform(nlpDF)
display(finalDf)

# COMMAND ----------

from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

trainDF, testDF = nlpDF.randomSplit([.8, .2], seed=42)

evaluator = MulticlassClassificationEvaluator(predictionCol='prediction', labelCol='label_index', metricName='f1')

paramGrid = (ParamGridBuilder()
            .addGrid(lr.regParam, [.01, .10])
            .build())

crossVal = CrossValidator(estimator=pipelineML, estimatorParamMaps=paramGrid, evaluator=evaluator, numFolds=3, seed=27)

crossValModel = crossVal.fit(trainDF)

print(crossValModel.avgMetrics)

# COMMAND ----------

predDF = crossValModel.transform(testDF)
display(predDF)

# COMMAND ----------

# MAGIC %md 
# MAGIC #Step 4) Collecting Metrics on test dataset

# COMMAND ----------

from pyspark.ml.evaluation import MulticlassClassificationEvaluator

evaluator = MulticlassClassificationEvaluator(metricName="f1", labelCol="label_index")
metricsDF = spark.createDataFrame([("f1", evaluator.evaluate(predDF)), 
                                   ("accuracy", evaluator.setMetricName("accuracy").evaluate(predDF))], ["Metric", "Value"])
display(metricsDF)

# COMMAND ----------

from datetime import date
today = date.today()
# dd/mm/YY
d1 = today.strftime("%d-%m-%Y")

# COMMAND ----------

import mlflow  
import mlflow.tracking
import mlflow.spark
#experiment_id = mlflow.create_experiment("/Users/debu.sinha@databricks.com/webinar-modeldrift/spam-review-classifier-model-registry")
experiment_id = mlflow.create_experiment("/Shared/experiments/spam-review-classifier")



#create a new run to log my best ML model
with mlflow.start_run(experiment_id=experiment_id, run_name="bestrun_"+d1) as run:
    # Log mlflow attributes for mlflow UI\
    
    mlflow.log_metric("f1", evaluator.evaluate(predDF))
    mlflow.log_metric("accuracy", evaluator.setMetricName("accuracy").evaluate(predDF))
    
    mlflow.spark.log_model(etlPipelineModel, "etlPipelineModel")
    mlflow.spark.log_model(nlpPipelineModel, "nlpPipelineModel")
    mlflow.spark.log_model(crossValModel.bestModel, "bestScoringModel")
    
    print("Inside MLflow Run with id %s" % run.info.run_uuid)


# COMMAND ----------

#experiment_id = mlflow.create_experiment("/Users/debu.sinha@databricks.com/webinar-modeldrift/spam-review-classifier-runs")
experiment_id = mlflow.create_experiment("/Shared/experiments/spam-review-classifier-NLP")
#create a new run to log my best ML model
with mlflow.start_run(experiment_id=experiment_id, run_name=d1) as run:
    # Log mlflow attributes for mlflow UI\
    
    mlflow.log_metric("f1", evaluator.evaluate(predDF))
    mlflow.log_metric("accuracy", evaluator.setMetricName("accuracy").evaluate(predDF))
