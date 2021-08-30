# Databricks notebook source
# MAGIC %md ---
# MAGIC title: Deceptive Opinion Spam filter using John Snow Labs
# MAGIC - Debu Sinha
# MAGIC tags:
# MAGIC - python
# MAGIC - spark-streaming
# MAGIC - NLP
# MAGIC - mlflow
# MAGIC - delta
# MAGIC - John Snow Labs
# MAGIC 
# MAGIC created_at: 2020-02-12
# MAGIC updated-on: 2021-08-28
# MAGIC tldr: Create a Spam review classifier using John Snow Labs on Databricks.
# MAGIC ---

# COMMAND ----------

# MAGIC %md
# MAGIC Deceptive Opinion Spam Corpus v1.4
# MAGIC ==================================
# MAGIC 
# MAGIC Overview
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
# MAGIC 
# MAGIC License
# MAGIC -------
# MAGIC 
# MAGIC This work is licensed under the Creative Commons
# MAGIC Attribution-NonCommercial-ShareAlike 3.0 Unported License. To view a copy of
# MAGIC this license, visit http://creativecommons.org/licenses/by-nc-sa/3.0/ or send a
# MAGIC letter to Creative Commons, 444 Castro Street, Suite 900, Mountain View,
# MAGIC California, 94041, USA.
# MAGIC 
# MAGIC If you use any of this data in your work, please cite the appropriate associated
# MAGIC paper (described above).

# COMMAND ----------

# MAGIC %md
# MAGIC # Step 0) Data Prep

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC all data is in the common location : /mnt/databricks-datasets-private/ML/NLP_JohnSnowLabs_FakeReviews

# COMMAND ----------

# MAGIC %sql
# MAGIC create database if not exists spam_reviews;
# MAGIC use spam_reviews;

# COMMAND ----------

# MAGIC %sql
# MAGIC create EXTERNAL TABLE IF NOT EXISTS spam_reviews_raw(review string) 
# MAGIC COMMENT 'This table has raw data for the spamreview_demo' 
# MAGIC PARTITIONED  BY(reviewpolarity string, label string, source string)  
# MAGIC LOCATION '/mnt/databricks-datasets-private/ML/NLP_JohnSnowLabs_FakeReviews/op_spam/';

# COMMAND ----------

# MAGIC %sql
# MAGIC msck repair table spam_reviews_raw;

# COMMAND ----------

# MAGIC %sql
# MAGIC create table if not exists spam_reviews.spam_reviews_delta_bronze
# MAGIC using delta
# MAGIC as select row_number() over(order by reviewpolarity, label, source) as sno, review, reviewpolarity, label, source from spam_reviews.spam_reviews_raw;

# COMMAND ----------

# MAGIC %md
# MAGIC # Step 1) Load the data

# COMMAND ----------

data = spark.sql("select sno, review, label from spam_reviews.spam_reviews_delta_bronze where not (reviewpolarity='positive' and label='fake')");

# COMMAND ----------

display(data)

# COMMAND ----------

data.count()

# COMMAND ----------

# MAGIC %md
# MAGIC # Step 2) Data preparation
# MAGIC * Label Encoding
# MAGIC * Normalization of the review sentences.
# MAGIC * Bagging the words.

# COMMAND ----------

from pyspark.ml import Pipeline
from pyspark.ml.feature import *
from pyspark.ml.classification import NaiveBayes
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.feature import OneHotEncoderEstimator, RegexTokenizer, CountVectorizer, IDF 
from pyspark.ml.classification import LogisticRegression

from pyspark.sql.functions import *
from sparknlp.annotator import *
from sparknlp.common import *
from sparknlp.base import *
import sparknlp

# COMMAND ----------

# MAGIC %md
# MAGIC Convert the categorical columns to numeric

# COMMAND ----------

string_col_list = [i[0] for i in data.dtypes if ((i[1] == 'string') and (i[0] != "review"))]
string_indexers = [StringIndexer(inputCol = categorical_col, outputCol = categorical_col + "_index") for categorical_col in string_col_list]
encoders = [OneHotEncoderEstimator(inputCols=[categorical_col+"_index"], outputCols=[categorical_col+"_vec"]) for categorical_col in string_col_list if categorical_col != "label"]

# COMMAND ----------

stages = string_indexers + encoders
indexer = Pipeline(stages=stages)
etlPipelineModel = indexer.fit(data)
indexed = etlPipelineModel.transform(data).drop("label")
display(indexed)

# COMMAND ----------

# MAGIC %md
# MAGIC Convert the input reviews into an array of lemmatize tokens.

# COMMAND ----------

document_assembler = DocumentAssembler().setInputCol("review").setOutputCol("document")
token_assembler = Tokenizer().setInputCols(["document"]).setOutputCol("token").setTargetPattern("\w+\'?\w+") # Tokenize
normalizer = Normalizer().setInputCols(["token"]).setOutputCol("normalized") # 
lemmatizer_model = LemmatizerModel.pretrained().setInputCols(["normalized"]).setOutputCol("lemmatized") # Lemmatize
finisher = Finisher().setInputCols(["lemmatized"]).setOutputAsArray(True) # Convert to Array

nlpPipeline = Pipeline(stages=[document_assembler, token_assembler, normalizer, lemmatizer_model, finisher])
nlpPipelineModel = nlpPipeline.fit(indexed)
nlpDF = nlpPipelineModel.transform(indexed)
display(nlpDF)

# COMMAND ----------

# MAGIC %md 
# MAGIC # Step 3) Feature engineering and trainings our machine learning model

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

# COMMAND ----------


