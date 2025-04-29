CREATE KEYSPACE IF NOT EXISTS reddit_sentiment 
WITH replication = {
    'class': 'NetworkTopologyStrategy', 
    'datacenter1': 3
} AND durable_writes = true;

USE reddit_sentiment;

CREATE TABLE IF NOT EXISTS reddit_data (
    subreddit TEXT,
    bucket INT,  // Hour of day (0-23)
    id TEXT,
    created_at TIMESTAMP,
    title TEXT,
    text TEXT,
    cleaned_text TEXT,
    type TEXT,
    sentiment_label TEXT,
    sentiment_score DECIMAL,
    insertion_time TIMESTAMP DEFAULT toTimestamp(now()),
    PRIMARY KEY ((subreddit, bucket), created_at, id)
) WITH bloom_filter_fp_chance = 0.01
    AND caching = {
        'keys': 'ALL', 
        'rows_per_partition': '100'
    }
    AND compaction = {
        'class': 'TimeWindowCompactionStrategy',
        'compaction_window_size': '1',
        'compaction_window_unit': 'HOURS'
    }
    AND compression = {
        'chunk_length_in_kb': '64',
        'class': 'LZ4Compressor'
    }
    AND default_time_to_live = 2592000  // 30 days
    AND speculative_retry = '99PERCENTILE';

// Create all recommended materialized views and indexes here