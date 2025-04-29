# Real-Time Sentiment Analysis of Reddit Comments

This project is a real-time big data pipeline that performs **sentiment analysis** on **Reddit comments**, using a distributed tech stack including **Apache Kafka**, **Apache Spark**, **Hugging Face Transformers**, **Cassandra**, and **Streamlit**. Everything is containerized with **Docker** for seamless deployment.

## Project Features

- Real-time data ingestion from Reddit using Reddit API (PRAW)
- Kafka-based streaming infrastructure
- Apache Spark Structured Streaming for real-time processing
- Sentiment analysis using pre-trained **RoBERTa (Twitter model)** from Hugging Face
- Storage of results in **Apache Cassandra**
- Interactive dashboard with **Streamlit**
- Fully containerized with **Docker Compose**

## Tech Stack

| Component          | Technology                          |
|--------------------|-------------------------------------|
| Data Source        | Reddit API (PRAW)                   |
| Messaging Queue    | Apache Kafka                        |
| Coordination       | Apache Zookeeper                    |
| Streaming Engine   | Apache Spark                        |
| Sentiment Model    | Hugging Face Transformers (RoBERTa) |
| Storage            | Apache Cassandra                    |
| Dashboard          | Streamlit                           |
| Containers         | Docker + Docker Compose             |

## Architecture

![Project-Architecture](https://github.com/user-attachments/assets/66841c24-14d5-4c2d-9a66-a64a46eed743)

## Directory Structure
```
  .
  ├── docker-compose.yml
  ├── README.md
  ├── cassandra/
  │   └── init.cql
  ├── consumer/
  │   ├── consumer.py
  │   ├── Dockerfile
  │   └── requirements.txt
  ├── dashboard/
  │   ├── dashboard.py
  │   ├── Dockerfile
  │   └── requirements.txt
  └── producer/
      ├── Dockerfile
      ├── producer.py
      └── requirements.txt
```


          
