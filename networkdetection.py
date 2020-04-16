import re

from influxdb import InfluxDBClient
from numpy.ma import sqrt
# from pyspark import SparkContext, SparkConf
from pyspark import SparkContext, SparkConf
from pyspark.mllib.clustering import KMeans
from pyspark.sql import SparkSession
from pyspark.streaming import StreamingContext
from pyspark.streaming.kafka import KafkaUtils

training_log_file_path = '/home/shubham/Downloads/bigdata/anomaly-detection/ML network detection/test_access_log.txt'
streaming_log_folder = 'file:///home/shubham/logfolder/'
conf = SparkConf().setAppName("Network Log Analyser").setMaster("spark://shubham-thinkpad:7077")
conf.set("spark.io.compression.codec", "snappy")
sc = SparkContext.getOrCreate(conf=conf)
sc.setLogLevel("INFO")
ssc = StreamingContext(sc, 10)
ssc.checkpoint('ckpt')
topic = "apache_nasa"
zk = "54.175.219.57:2181"
brokers = "54.175.219.57:9092,54.175.219.57:9093,54.175.219.57:9094"

TRAINING_APACHE_ACCESS_LOG_PATTERN = "(?P<source>.+?) \[\d+:\d+:\d+:\d+\] \".+?\" (?P<response_code>\d+) .+?"
APACHE_ACCESS_LOG_PATTERN = '^(\S+) (\S+) (\S+) \[([\w:/]+\s[+\-]\d{4})\] "(\S+) (\S+)\s*(\S*)\s*" (\d{3}) (\S+)'


def parse_training_line(logline, pattern):
    line = re.match(pattern, logline)

    if line:
        logline = line.groupdict()

    nxx = 0
    twoxx = 0

    # If response code is 2xx, then we set the count for twoxx as 1
    if str(logline['response_code']).startswith("2"):
        twoxx = 1
    # If response code is 5xx, then we set the count for nxx is 1
    elif str(logline['response_code']).startswith("5"):
        nxx = 1

    # return data in the form (1.1.1.1, [1, 0])
    return (logline['source'], [twoxx, nxx])


def parse_log_line(log_line):
    line = re.search(APACHE_ACCESS_LOG_PATTERN, log_line)

    nxx = 0
    twoxx = 0
    # If response code is 2xx, then we set the count for twoxx as 1
    if str(line.group(8)).startswith("2"):
        twoxx = 1
    # If response code is 5xx, then we set the count for nxx is 1
    elif str(line.group(8)).startswith("5"):
        nxx = 1

    # return data in the form (1.1.1.1, [1, 0])
    return line.group(1), [twoxx, nxx]


def extract_features(val1, val2):
    # add up counts of 2xx and 5xx for each of the source IP
    # val1 is for 1st record and val2 for second record of same source ip. likewise until all the records are added
    twoxx = val1[0] + val2[0]
    nxx = val1[1] + val2[1]

    return [twoxx, nxx]


# Cluster Prediction and distance calculation
def predict_cluster(row, model):
    # Predict the cluster for the current data row
    cluster = model.predict(row[1])

    # Find the center for the current cluster
    cluster_center = model.centers[cluster]

    # Calculate the disance between the Current Row Data and Cluster Center
    distance = sqrt(sum([x ** 2 for x in (row[1] - cluster_center)]))

    # return (row[0], distance, {"cluster": model.predict(row[1]), "twoxx": row[1][0], "nxx": row[1][1]})
    return row[0], distance, model.predict(row[1]), row[1][0], row[1][1]


def getSparkSessionInstance(sparkConf):
    if "sparkSessionSingletonInstance" not in globals():
        globals()["sparkSessionSingletonInstance"] = SparkSession \
            .builder \
            .config(conf=sparkConf) \
            .getOrCreate()
    return globals()["sparkSessionSingletonInstance"]


def sendRecord(tup):
    events = [{
        "measurement": "distance",
        "tags": {
            "source": tup[0],
            "cluster": tup[2]
        },
        "fields":
            {
                "deviation": tup[1],
                "2xx": tup[3],
                "4xx": tup[4]
            }
    }]

    dbclient = InfluxDBClient('54.175.219.57', 8086, 'root', 'root', 'example')
    dbclient.write_points(events)


trainingData = sc.textFile(training_log_file_path)
rawTrainingData = trainingData.map(lambda s: parse_training_line(s, TRAINING_APACHE_ACCESS_LOG_PATTERN)).cache()
print('\ntotal training lines : ', rawTrainingData.count())

# process the rawTrainingdata RDD toeach hostkey using mapreducebykey. mapreduceBykey returns KV.
# Reduce returns a single value result contains count of 2xx and 3x against an IP
rawTrainingData = rawTrainingData.reduceByKey(extract_features)
print("training dataset after reduce: ", rawTrainingData.collect())
print('total training lines after reduce by key : ', rawTrainingData.count())

# K-means accepts data in the form of [a, b] its called feature vector. use vector assembler or map function.
# Converts to map of count of 2xx and 3xx
training_dataset = rawTrainingData.map(lambda data: data[1])
print("TRAINING DATASET for Kmean cluster: ", training_dataset.collect())
print('total training lines after reformat : ', rawTrainingData.count())

# set cluster count equals to 2
cluster_count = 2

# train the k-means algo to get the model
trained_model = KMeans.train(training_dataset, cluster_count)

# print the cluster centroids from trained model
for center in range(cluster_count):
    print('centre ', center, trained_model.centers[center])

    # streamingData = KafkaUtils.createStream(ssc, "localhost:2181", "test-consumer-group", {"test" : 1})
    # lines = streamingData.map(lambda x:x[1])

# df.selectExpr("CAST(key AS STRING)", "CAST(value AS STRING)")

stream_data_init = KafkaUtils.createDirectStream(ssc, [topic], {"metadata.broker.list": brokers})
# stream_data_init = KafkaUtils.createStream(
#    ssc,
#    zk,
#    'consumer-group-name',
#    {topic: 1}
# )
print(type(stream_data_init))
stream_data = stream_data_init.map(
    lambda s: parse_log_line(s[1])).reduceByKeyAndWindow(extract_features,
                                                            lambda a, b: [a[0] - b[0],
                                                                          a[1] - b[1]], 60,
                                                            60).map(
    lambda s: predict_cluster(s, trained_model))

# stream_data = ssc.textFileStream(streaming_log_folder)\
#    .map(
#    lambda s: parse_log_line(s)).reduceByKeyAndWindow(extract_features,
#                                                      lambda a, b: [a[0] - b[0],
#                                                                    a[1] - b[1]], 60,
#                                                      60).map(
#    lambda s: predict_cluster(s, trained_model))

# map( lambda x: x.split(" "))
# map(lambda s: parse_log_line(s, APACHE_ACCESS_LOG_PATTERN))

stream_data.pprint()
stream_data.foreachRDD(lambda rdd: rdd.foreach(sendRecord))
print(stream_data.count())
print("starting")

ssc.start()
ssc.awaitTermination()
sc.stop()
