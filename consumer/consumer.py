import os
import json
import time
import re
from dotenv import load_dotenv
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, from_json, udf, when, concat_ws
from pyspark.sql.types import StructType, StructField, StringType, DoubleType, LongType
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from cassandra.cluster import Cluster
from cassandra.auth import PlainTextAuthProvider
from cassandra.query import SimpleStatement

# Load environment variables
load_dotenv()

# Configuration with better error handling
def get_config():
    return {
        'kafka': {
            'topic': os.getenv('KAFKA_TOPIC', 'reddit_posts'),
            'bootstrap_servers': os.getenv('KAFKA_BOOTSTRAP_SERVERS', 'kafka:9092')
        },
        'cassandra': {
            'host': os.getenv('CASSANDRA_HOST', 'cassandra'),
            'port': int(os.getenv('CASSANDRA_PORT', '9042')),
            'keyspace': os.getenv('CASSANDRA_KEYSPACE', 'reddit_sentiment')
        }
    }

def connect_to_cassandra(max_retries=30, retry_delay=10):
    """Improved Cassandra connection with better error handling"""
    config = get_config()['cassandra']
    
    for attempt in range(1, max_retries + 1):
        try:
            print(f"Connecting to Cassandra (attempt {attempt}/{max_retries})...")
            cluster = Cluster([config['host']], port=config['port'])
            session = cluster.connect()
            
            # Check and create keyspace if needed
            rows = session.execute("SELECT keyspace_name FROM system_schema.keyspaces")
            keyspaces = {row.keyspace_name for row in rows}
            
            if config['keyspace'] not in keyspaces:
                print(f"Creating keyspace {config['keyspace']}...")
                session.execute(f"""
                    CREATE KEYSPACE IF NOT EXISTS {config['keyspace']}
                    WITH replication = {{'class': 'SimpleStrategy', 'replication_factor': '1'}}
                """)
            
            session.set_keyspace(config['keyspace'])
            
            # Create table with all required columns
            session.execute("""
                CREATE TABLE IF NOT EXISTS reddit_data (
                    id TEXT PRIMARY KEY,
                    created_at BIGINT,
                    subreddit TEXT,
                    title TEXT,
                    text TEXT,
                    type TEXT,
                    sentiment_label TEXT,
                    sentiment_score DOUBLE,
                    cleaned_text TEXT
                )
            """)
            
            # Create indexes if they don't exist
            for column in ['subreddit', 'created_at', 'sentiment_label']:
                try:
                    session.execute(f"CREATE INDEX IF NOT EXISTS ON reddit_data ({column})")
                except Exception as e:
                    if "already exists" not in str(e):
                        print(f"Error creating index on {column}: {e}")
            
            print(f"Successfully connected to Cassandra keyspace '{config['keyspace']}'")
            return cluster, session
            
        except Exception as e:
            print(f"Attempt {attempt} failed: {str(e)}")
            if attempt < max_retries:
                time.sleep(retry_delay)
    
    print("Failed to connect to Cassandra after maximum attempts")
    return None, None

def create_spark_session():
    """Create and configure Spark session with better resource management"""
    try:
        config = get_config()
        
        spark = SparkSession.builder \
            .appName("RedditSentimentAnalysis") \
            .config("spark.jars.packages", 
                   "org.apache.spark:spark-sql-kafka-0-10_2.12:3.3.0,"
                   "com.datastax.spark:spark-cassandra-connector_2.12:3.2.0") \
            .config("spark.cassandra.connection.host", config['cassandra']['host']) \
            .config("spark.cassandra.connection.port", config['cassandra']['port']) \
            .config("spark.sql.shuffle.partitions", "4")\
            .config("spark.default.parallelism", "4") \
            .getOrCreate()
        
        spark.sparkContext.setLogLevel("ERROR")
        return spark
    except Exception as e:
        print(f"Failed to create Spark session: {e}")
        raise

def preprocess_text(text):
    """Twitter-Roberta specific text preprocessing"""
    if not text or not isinstance(text, str):
        return ""
    
    # Twitter-Roberta specific preprocessing
    new_text = []
    for t in text.split(" "):
        t = '@user' if t.startswith('@') and len(t) > 1 else t
        t = 'http' if t.startswith('http') else t
        new_text.append(t)
    return " ".join(new_text)

def initialize_sentiment_analyzer():
    """Initialize Twitter-Roberta sentiment analysis pipeline"""
    try:
        model_name = "cardiffnlp/twitter-roberta-base-sentiment-latest"
        print(f"Loading Twitter-Roberta sentiment analysis model: {model_name}")
        
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
        
        return pipeline(
            "sentiment-analysis",
            model=model,
            tokenizer=tokenizer,
            device=0 if os.environ.get('CUDA_VISIBLE_DEVICES') else -1,
            return_all_scores=False
        )
    except Exception as e:
        print(f"Failed to initialize sentiment analyzer: {e}")
        raise

def analyze_sentiment(text, sentiment_analyzer):
    """Sentiment analysis with Twitter-Roberta model"""
    try:
        if not text or not isinstance(text, str) or len(text.strip()) < 3:
            return ("neutral", 0.5, "")
        
        # Apply Twitter-Roberta specific preprocessing
        processed_text = preprocess_text(text)
        
        if not processed_text:
            return ("neutral", 0.5, "")
        
        # Handle long texts
        if len(processed_text) > 512:
            chunks = [processed_text[i:i+512] for i in range(0, len(processed_text), 512)]
            results = []
            for chunk in chunks[:3]:  # Limit to first 3 chunks
                result = sentiment_analyzer(chunk)[0]
                results.append((result['label'].lower(), float(result['score'])))
            
            if not results:
                return ("neutral", 0.5, processed_text[:512] + "... [truncated]")
            
            # Take the most confident prediction
            results.sort(key=lambda x: x[1], reverse=True)
            label, score = results[0]
            
            return (label, score, processed_text[:512] + "... [truncated]")
        
        # Process normal length text
        result = sentiment_analyzer(processed_text[:512])[0]
        label = result['label'].lower()
        score = float(result['score'])
        
        return (label, score, processed_text)
        
    except Exception as e:
        print(f"Sentiment analysis error: {str(e)}")
        return ("neutral", 0.5, processed_text if 'processed_text' in locals() else "")

def process_data(spark, cluster, session):
    """Main data processing function with improved error handling"""
    config = get_config()
    
    # First verify Kafka connection
    try:
        from kafka import KafkaConsumer
        consumer = KafkaConsumer(
            bootstrap_servers=config['kafka']['bootstrap_servers'],
            request_timeout_ms=10000
        )
        consumer.topics()
        consumer.close()
        print("Successfully connected to Kafka")
    except Exception as e:
        print(f"Failed to connect to Kafka: {e}")
        return

    # Define schema
    schema = StructType([
        StructField("id", StringType(), nullable=True),
        StructField("created_at", LongType(), nullable=True),
        StructField("subreddit", StringType(), nullable=True),
        StructField("title", StringType(), nullable=True),
        StructField("text", StringType(), nullable=True),
        StructField("type", StringType(), nullable=True)
    ])
    
    # Initialize sentiment analyzer
    try:
        print("Loading Twitter-Roberta sentiment analysis model...")
        sentiment_analyzer = initialize_sentiment_analyzer()
        
        # Test the model
        test_result = sentiment_analyzer("This is great!")[0]
        print(f"Model test successful: {test_result}")
    except Exception as e:
        print(f"Failed to initialize sentiment analyzer: {e}")
        return

    # Register UDFs
    preprocess_text_udf = udf(preprocess_text, StringType())
    
    def analyze_sentiment_udf_wrapper(text):
        result = analyze_sentiment(text, sentiment_analyzer)
        return json.dumps(result)
    
    analyze_sentiment_udf = udf(analyze_sentiment_udf_wrapper, StringType())

    # Create checkpoint directory
    checkpoint_dir = "/tmp/checkpoints"
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
        print(f"Created checkpoint directory at {checkpoint_dir}")

    # Kafka stream processing
    try:
        print(f"Starting Kafka stream from {config['kafka']['bootstrap_servers']}...")
        
        # Read from Kafka
        df = spark \
            .readStream \
            .format("kafka") \
            .option("kafka.bootstrap.servers", config['kafka']['bootstrap_servers']) \
            .option("subscribe", config['kafka']['topic']) \
            .option("startingOffsets", "earliest") \
            .option("failOnDataLoss", "false") \
            .option("maxOffsetsPerTrigger", "100") \
            .load()
        
        # Parse JSON
        parsed_df = df \
            .selectExpr("CAST(value AS STRING) as json", "CAST(key AS STRING) as key") \
            .withColumn("data", from_json(col("json"), schema))
        
        # Extract fields
        extracted_df = parsed_df \
            .select(
                col("data.id").alias("id"),
                col("data.created_at").alias("created_at"),
                col("data.subreddit").alias("subreddit"),
                col("data.title").alias("title"),
                col("data.text").alias("text"),
                col("data.type").alias("type")
            ) \
            .filter(col("id").isNotNull())
        
        # Apply sentiment analysis
        sentiment_df = extracted_df \
            .withColumn("title_safe", when(col("title").isNotNull(), col("title")).otherwise("")) \
            .withColumn("text_safe", when(col("text").isNotNull(), col("text")).otherwise("")) \
            .withColumn("combined_text", concat_ws(" ", col("title_safe"), col("text_safe"))) \
            .withColumn("sentiment_result", analyze_sentiment_udf(col("combined_text"))) \
            .drop("combined_text", "title_safe", "text_safe")
        
        # Prepare Cassandra statement
        insert_stmt = session.prepare("""
            INSERT INTO reddit_data (
                id, created_at, subreddit, title, text, type,
                sentiment_label, sentiment_score, cleaned_text
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """)
        
        def process_batch(batch_df, batch_id):
            """Process each batch of data"""
            count = batch_df.count()
            print(f"Processing batch {batch_id} with {count} rows")
            
            if count == 0:
                return
            
            rows = batch_df.collect()
            successful = 0
            
            for row in rows:
                try:
                    sentiment_result = json.loads(row["sentiment_result"])
                    if len(sentiment_result) != 3:
                        continue
                        
                    sentiment_label, sentiment_score, processed_text = sentiment_result
                    
                    session.execute(insert_stmt, (
                        row["id"],
                        row["created_at"],
                        row["subreddit"],
                        row["title"],
                        row["text"],
                        row["type"],
                        sentiment_label,
                        sentiment_score,
                        processed_text
                    ))
                    successful += 1
                    
                except Exception as e:
                    print(f"Error processing row {row.get('id', 'unknown')}: {str(e)}")
            
            print(f"Processed batch {batch_id}: {successful}/{len(rows)} successful")
        
        # Start streaming
        query = sentiment_df \
            .writeStream \
            .foreachBatch(process_batch) \
            .outputMode("append") \
            .option("checkpointLocation", checkpoint_dir) \
            .trigger(processingTime="5 seconds") \
            .start()
        
        print("Streaming query started successfully")
        
        try:
            while query.isActive:
                print(f"Stream status: {query.status}")
                if query.lastProgress:
                    print(f"Recent progress: {query.lastProgress}")
                time.sleep(10)
                
            query.awaitTermination()
        except Exception as e:
            print(f"Error during stream monitoring: {e}")
            if query and query.isActive:
                query.stop()
                print("Query stopped due to error")
        
    except Exception as e:
        print(f"Stream processing failed: {e}")
        raise

def main():
    """Main function with proper resource management"""
    cluster, session = None, None
    spark = None
    
    try:
        # Connect to Cassandra
        cluster, session = connect_to_cassandra()
        if not session:
            print("Failed to establish Cassandra connection")
            return
        
        # Create Spark session
        spark = create_spark_session()
        
        # Process data
        print("Starting data processing...")
        process_data(spark, cluster, session)
        
    except KeyboardInterrupt:
        print("Process interrupted by user")
    except Exception as e:
        print(f"Fatal error: {e}")
    finally:
        # Cleanup resources
        if spark:
            try:
                spark.stop()
                print("Spark session closed")
            except Exception as e:
                print(f"Error stopping Spark: {e}")
        
        if cluster:
            try:
                cluster.shutdown()
                print("Cassandra connection closed")
            except Exception as e:
                print(f"Error closing Cassandra connection: {e}")

if __name__ == "__main__":
    main()