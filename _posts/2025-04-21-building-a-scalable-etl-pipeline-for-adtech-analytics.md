---
layout: post
title: 'Building a Scalable ETL Pipeline for AdTech Analytics'
description: 
image: 
category: [Data Engineering]
tags: python postgres etl clickhouse 
date: 2025-04-21 17:47 +0200
---

In the world of digital advertising, data is everything. Transforming raw operational data into actionable insights requires robust analytics pipelines. Recently, I implemented a comprehensive ETL (Extract, Transform, Load) solution for an advertising platform that moves data from PostgreSQL to ClickHouse for high-performance analytics. Let me walk you through the process, design decisions, and implementation details.

## Overview of the Challenge

The challenge was to build a pipeline that would:

1. Extract campaign, impression, and click data from a PostgreSQL operational database
2. Transform the data into an analytics-friendly format
3. Load it into ClickHouse, a columnar DBMS optimized for analytical queries
4. Create materialized views for key advertising KPIs

The solution needed to handle both initial data loads and incremental updates, with robust error handling and monitoring capabilities.

## Architecture Design

After reviewing the requirements, I designed the following architecture:

```
PostgreSQL (Source) → ETL Pipeline → ClickHouse (Target)
```

### Source Database (PostgreSQL)

The operational database contained four primary tables:
- `advertiser`: Information about companies running ad campaigns
- `campaign`: Campaign configurations with bid amounts and budgets
- `impressions`: Records of ads being displayed
- `clicks`: Records of users clicking on ads

### Target Schema (ClickHouse)

For the analytical layer, I designed a dimensional model with:

- Dimension tables:
  - `dim_advertiser`
  - `dim_campaign`
- Fact tables:
  - `fact_impressions`
  - `fact_clicks`
- Materialized views for KPIs:
  - Campaign CTR (Click-Through Rate)
  - Daily performance metrics
  - Campaign daily performance
  - Cost efficiency metrics
  - Advertiser performance overviews

## Implementation Details

### 1. Setting Up the Core Components

I implemented the ETL pipeline in Python 3.12 with a modular design to separate concerns:

```python
from .config import AppConfig, PostgresConfig, ClickhouseConfig, ETLConfig
from .db import PostgresConnector, ClickhouseConnector
from .schema import SchemaManager
from .pipeline import (
    DataExtractor, DataTransformer, DataLoader, ETLPipeline
)
```

### 2. Database Connectors

The first components I built were the database connectors, which encapsulate connection management and query execution:

```python
class PostgresConnector:
    """PostgreSQL connection manager."""
    
    def __init__(self, config: PostgresConfig):
        """Initialize with PostgreSQL configuration."""
        self.config = config
        self.conn = None
    
    def connect(self) -> bool:
        """Establish connection to PostgreSQL database."""
        try:
            self.conn = psycopg.connect(
                self.config.connection_string,
                autocommit=False,
            )
            logger.info("Connected to PostgreSQL database")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to PostgreSQL: {e}")
            return False
    
    # Additional methods for query execution, etc.
```

Similarly, I implemented a `ClickhouseConnector` for managing ClickHouse connections:

```python
class ClickhouseConnector:
    """ClickHouse connection manager."""
    
    def __init__(self, config: ClickhouseConfig):
        """Initialize with ClickHouse configuration."""
        self.config = config
        self.client = None
    
    # Methods for connection, query execution, etc.
```

### 3. Schema Management

I created a `SchemaManager` class to handle ClickHouse schema setup and updates:

```python
class SchemaManager:
    """Manages ClickHouse schema creation and updates."""
    
    def __init__(self, db_connector: ClickhouseConnector, config: ETLConfig):
        """Initialize with ClickHouse connector and configuration."""
        self.db = db_connector
        self.config = config
    
    def setup_schema(self) -> bool:
        """Initialize ClickHouse schema if not exists."""
        try:
            self.db.execute_file(self.config.schema_path)
            logger.info("ClickHouse schema initialized successfully")
            return True
        except Exception as e:
            logger.error(f"Error setting up ClickHouse schema: {e}")
            return False
```

### 4. The ETL Pipeline Components

The core of the implementation consists of three main components:

#### A. Data Extractor

```python
class DataExtractor:
    """Extracts data from PostgreSQL source database."""
    
    def __init__(self, db: PostgresConnector):
        """Initialize with PostgreSQL connector."""
        self.db = db
    
    def extract_advertisers(self, since: datetime) -> List[Tuple]:
        """Extract advertisers updated since the given timestamp."""
        query = """
            SELECT id, name, updated_at, created_at 
            FROM advertiser
            WHERE updated_at > %s OR created_at > %s
        """
        return self.db.execute_query(query, (since, since))
    
    # Additional methods for extracting campaigns, impressions, clicks
```

#### B. Data Transformer

```python
class DataTransformer:
    """Transforms data for loading into ClickHouse."""
    
    @staticmethod
    def transform_advertisers(rows: List[Tuple]) -> List[Tuple]:
        """Transform advertiser data for ClickHouse."""
        transformed = []
        for adv_id, name, updated_at, created_at in rows:
            transformed.append((
                adv_id,
                name,
                updated_at or datetime.now(),
                created_at or datetime.now()
            ))
        return transformed
    
    # Additional transformation methods
```

#### C. Data Loader

```python
class DataLoader:
    """Loads transformed data into ClickHouse."""
    
    def __init__(self, db: ClickhouseConnector):
        """Initialize with ClickHouse connector."""
        self.db = db
    
    def load_advertisers(self, data: List[Tuple]) -> int:
        """Load advertiser data into ClickHouse."""
        if not data:
            return 0
            
        query = """
            INSERT INTO analytics.dim_advertiser
            (advertiser_id, name, updated_at, created_at)
            VALUES
        """
        self.db.execute_query(query, data)
        return len(data)
    
    # Additional loading methods
```

### 5. Orchestrating the ETL Process

I created an `ETLPipeline` class to orchestrate the entire process:

```python
class ETLPipeline:
    """Main ETL pipeline that orchestrates extract, transform, and load."""
    
    def __init__(
        self, 
        extractor: DataExtractor, 
        transformer: DataTransformer, 
        loader: DataLoader
    ):
        """Initialize with extractor, transformer, and loader components."""
        self.extractor = extractor
        self.transformer = transformer
        self.loader = loader
        self.last_sync = {
            'advertiser': datetime.min,
            'campaign': datetime.min,
            'impressions': datetime.min,
            'clicks': datetime.min
        }
        
        # Tracking for sync statistics
        self.sync_stats = {
            'advertiser': 0,
            'campaign': 0,
            'impressions': 0,
            'clicks': 0
        }
    
    def run_sync_cycle(self) -> bool:
        """Run a complete ETL cycle."""
        try:
            logger.info("Starting ETL sync cycle")
            
            # Reset sync statistics
            for key in self.sync_stats:
                self.sync_stats[key] = 0
            
            # Sync dimension tables first
            self.sync_stats['advertiser'] = self.sync_advertisers()
            self.sync_stats['campaign'] = self.sync_campaigns()
            
            # Then sync fact tables
            self.sync_stats['impressions'] = self.sync_impressions() 
            self.sync_stats['clicks'] = self.sync_clicks()
            
            # Log sync summary
            logger.info("ETL sync cycle completed successfully")
            logger.info(f"Sync summary: "
                       f"Advertisers: {self.sync_stats['advertiser']}, "
                       f"Campaigns: {self.sync_stats['campaign']}, "
                       f"Impressions: {self.sync_stats['impressions']}, "
                       f"Clicks: {self.sync_stats['clicks']}")
            
            return True
            
        except Exception as e:
            logger.error(f"ETL sync cycle failed: {e}")
            return False
```

### 6. Implementing Incremental Updates

One of the most critical aspects of the implementation was handling incremental updates efficiently. I designed the system to track the last sync timestamp for each entity:

```python
def sync_advertisers(self) -> int:
    """Sync advertisers from PostgreSQL to ClickHouse."""
    try:
        # Extract only data updated since last sync
        rows = self.extractor.extract_advertisers(self.last_sync['advertiser'])
        if not rows:
            logger.info("No new advertisers to sync")
            return 0
        
        # Transform and load
        data = self.transformer.transform_advertisers(rows)
        count = self.loader.load_advertisers(data)
        
        # Update the last sync timestamp
        latest_update = self.last_sync['advertiser']
        for _, _, updated_at, created_at in rows:
            if updated_at and updated_at > latest_update:
                latest_update = updated_at
            if created_at and created_at > latest_update:
                latest_update = created_at
        
        self.last_sync['advertiser'] = latest_update
        logger.info(f"Synced {count} advertisers")
        return count
        
    except Exception as e:
        logger.error(f"Error syncing advertisers: {e}")
        return 0
```

### 7. Main Service Loop

Finally, I implemented a main service class to tie everything together:

```python
def run_service(self, run_once: bool = False, interval: Optional[int] = None, force_full_sync: bool = False) -> None:
    """Run the ETL service continuously or once."""
    sync_interval = interval or self.config.etl.sync_interval
    
    if not self.initialize():
        self.logger.error("Service initialization failed. Exiting.")
        sys.exit(1)
    
    if force_full_sync:
        self.logger.info("Forcing full sync - resetting all sync timestamps")
        for key in self.etl_pipeline.last_sync:
            self.etl_pipeline.last_sync[key] = datetime.min
    
    self.logger.info(f"AdTech ETL service started with sync interval: {sync_interval}s")
    
    try:
        if run_once:
            success = self.run_sync()
            if not success:
                sys.exit(1)
        else:
            # Continuous operation
            while True:
                success = self.run_sync()
                if not success:
                    self.logger.warning(f"Waiting {sync_interval} seconds before retry...")
                
                self.logger.info(f"Sleeping for {sync_interval} seconds...")
                time.sleep(sync_interval)
                
    except KeyboardInterrupt:
        self.logger.info("ETL service interrupted, shutting down")
    except Exception as e:
        self.logger.critical(f"Unexpected error: {e}")
        sys.exit(1)
    finally:
        # Clean up resources
        if hasattr(self, 'pg_connector'):
            self.pg_connector.close()
        if hasattr(self, 'ch_connector'):
            self.ch_connector.close()
```

## ClickHouse Optimization

A key part of the solution was optimizing the ClickHouse schema for analytical queries:

```sql
-- Dimension tables with ReplacingMergeTree engine
CREATE TABLE IF NOT EXISTS analytics.dim_advertiser
(
    advertiser_id UInt32,
    name String,
    updated_at DateTime,
    created_at DateTime
) ENGINE = ReplacingMergeTree(updated_at)
ORDER BY (advertiser_id);

-- Fact tables with partitioning
CREATE TABLE IF NOT EXISTS analytics.fact_impressions
(
    impression_id UInt32,
    campaign_id UInt32,
    event_date Date,
    event_time DateTime,
    created_at DateTime
) ENGINE = MergeTree()
PARTITION BY toYYYYMM(event_date)
ORDER BY (campaign_id, event_date);
```

### Materialized Views for KPIs

I created several materialized views to pre-calculate common KPIs:

```sql
-- Materialized view for campaign CTR
CREATE MATERIALIZED VIEW IF NOT EXISTS analytics.mv_campaign_ctr
(
    campaign_id UInt32,
    campaign_name String,
    advertiser_name String,
    impressions UInt64,
    clicks UInt64,
    ctr Float64
)
ENGINE = SummingMergeTree()
ORDER BY (campaign_id)
POPULATE AS
SELECT
    c.campaign_id,
    c.name AS campaign_name,
    a.name AS advertiser_name,
    COUNT(DISTINCT i.impression_id) AS impressions,
    COUNT(DISTINCT cl.click_id) AS clicks,
    COUNT(DISTINCT cl.click_id) / COUNT(DISTINCT i.impression_id) AS ctr
FROM dim_campaign c
JOIN dim_advertiser a ON c.advertiser_id = a.advertiser_id
LEFT JOIN fact_impressions i ON c.campaign_id = i.campaign_id
LEFT JOIN fact_clicks cl ON c.campaign_id = cl.campaign_id
GROUP BY c.campaign_id, c.name, a.name;
```

## Testing

I implemented comprehensive testing with pytest to ensure the reliability of the ETL pipeline:

1. **Unit tests** for individual components
2. **Integration tests** for the end-to-end pipeline
3. **Schema tests** for database schema validation

For example, the unit tests for the Transformer component:

```python
@pytest.mark.unit
class TestDataTransformer:
    """Tests for DataTransformer."""

    def test_transform_advertisers(self):
        """Test transforming advertiser data."""
        now = datetime.now()
        input_data = [(1, 'Advertiser A', now, now)]
        
        transformer = DataTransformer()
        result = transformer.transform_advertisers(input_data)
        
        assert result == [(1, 'Advertiser A', now, now)]
```

## Deployment with Docker

I containerized the entire solution using Docker to ensure consistent operation:

```dockerfile
FROM python:3.12-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy ETL files
COPY . .

# Run the ETL script
CMD ["python", "main.py"]
```

And orchestrated the services with Docker Compose:

```yaml
services:
  # PostgreSQL
  postgres:
    image: postgres:17
    container_name: psql_source
    env_file: .env
    ports:
      - "6543:5432"
    volumes:
      - postgres_data:/var/lib/postgresql/data
    
  # ClickHouse
  clickhouse:
    image: clickhouse/clickhouse-server
    container_name: ch_analytics
    env_file: .env
    ports:
      - "8124:8123"
      - "9001:9000"
    volumes:
      - clickhouse_data:/var/lib/clickhouse
  
  # ETL Service
  etl:
    build:
      context: ./etl
      dockerfile: Dockerfile.etl
    container_name: adtech_etl
    restart: unless-stopped
    depends_on:
      postgres:
        condition: service_healthy
      clickhouse:
        condition: service_healthy
    env_file: .env
    volumes:
      - ./etl:/app
```

## CI/CD Pipeline

To ensure code quality, I set up a GitHub Actions workflow:

```yaml
name: CI

on:
  push:
    branches: [ main, dev ]
  pull_request:
    branches: [ main, dev ]

jobs:
  lint:
    name: Code Quality Checks
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      
      - name: Set up Python 3.12
        uses: actions/setup-python@v5
        with:
          python-version: '3.12'
      
      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install ruff==0.11.0
      
      - name: Lint with ruff
        run: python -m ruff check etl tests

  test:
    name: Unit Tests
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      
      - name: Set up Python 3.12
        uses: actions/setup-python@v5
        with:
          python-version: '3.12'
      
      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install -r etl/requirements.txt
          pip install pytest==8.3.5
      
      - name: Run unit tests
        run: python -m pytest -xvs -m "unit"
```

## Scalability Approaches

As data volumes grow in advertising platforms, the ETL pipeline must scale accordingly. Here are the key scalability approaches I implemented and recommend for further expansion:

### 1. Horizontal Scaling with Distributed Processing

The current architecture can be enhanced for horizontal scalability by implementing:

```python
class DistributedETLPipeline(ETLPipeline):
    """Distributed version of the ETL pipeline that supports partitioned processing."""
    
    def __init__(self, extractor, transformer, loader, partition_count=4):
        super().__init__(extractor, transformer, loader)
        self.partition_count = partition_count
        
    def partition_data(self, table_name, date_field, partition_key):
        """Create logical partitions for parallel processing."""
        partition_ranges = []
        # Calculate partition boundaries based on time or ID ranges
        return partition_ranges
        
    def run_partitioned_sync(self, table_name, executor):
        """Execute sync operations across multiple partitions in parallel."""
        partitions = self.partition_data(table_name, 'created_at', 'id')
        futures = []
        
        for partition in partitions:
            # Submit each partition for parallel execution
            future = executor.submit(self.sync_partition, table_name, partition)
            futures.append(future)
            
        # Gather results
        results = [future.result() for future in futures]
        return sum(results)
```

With this approach, we can use Python's `concurrent.futures` or distributed task queues like Celery to process partitions in parallel across multiple worker nodes.

### 2. Time-Based Batching for High-Volume Tables

For fact tables with millions of daily records (impressions, clicks), implementing time-based batching reduces memory pressure:

```python
def sync_impressions_with_batching(self, batch_size=10000, time_window_hours=1) -> int:
    """Sync impressions with time-based batching."""
    total_synced = 0
    current_time = self.last_sync['impressions']
    end_time = datetime.now()
    
    while current_time < end_time:
        # Calculate next batch window
        next_window = current_time + timedelta(hours=time_window_hours)
        if next_window > end_time:
            next_window = end_time
            
        # Extract and process just this time window
        rows = self.extractor.extract_impressions_by_window(current_time, next_window)
        if rows:
            data = self.transformer.transform_impressions(rows)
            count = self.loader.load_impressions(data)
            total_synced += count
            
        # Move to next window
        current_time = next_window
        
    # Update the final sync timestamp
    self.last_sync['impressions'] = end_time
    return total_synced
```

### 3. Database-Level Scaling

As the system scales, the database architecture can be enhanced:

#### PostgreSQL Scaling:
- Implement read replicas to isolate the ETL read load from operational writes
- Use logical replication with PostgreSQL's Change Data Capture (CDC) features
- Consider a replication-based CDC tool like Debezium for near real-time data streaming

#### ClickHouse Scaling:
- Implement a ClickHouse cluster with data sharding across multiple nodes
- Optimize the sharding key for commonly queried dimensions (e.g., by campaign_id)
- Implement distributed tables to abstract the sharding complexity:

```sql
-- Distributed table definition
CREATE TABLE IF NOT EXISTS analytics.dist_fact_impressions 
AS analytics.fact_impressions
ENGINE = Distributed(cluster_name, analytics, fact_impressions, rand());
```

### 4. Resilient Work Queue Architecture

For extreme scale, transition from a scheduled polling approach to an event-driven architecture:

```python
# In a message consumer service
def process_etl_message(self, message):
    """Process a message from the ETL work queue."""
    try:
        entity_type = message['entity_type']
        batch_id = message['batch_id']
        time_range = message.get('time_range', None)
        
        # Process the specific batch
        if entity_type == 'impressions':
            self.etl_pipeline.sync_impressions_batch(batch_id, time_range)
        elif entity_type == 'clicks':
            self.etl_pipeline.sync_clicks_batch(batch_id, time_range)
        # etc.
        
        # Acknowledge successful processing
        self.queue.acknowledge(message['id'])
    except Exception as e:
        # Failed processing - handle with retry logic
        self.queue.retry(message['id'])
        logger.error(f"Failed to process ETL message: {e}")
```

This approach works well with Apache Kafka, RabbitMQ, or cloud-native solutions like AWS SQS/SNS for truly decoupled processing.

## Optimization Techniques

Beyond the initial implementation, I've identified several optimization opportunities:

### 1. Query Optimization for Extraction

Improving extraction performance through optimized queries:

```python
def extract_impressions_optimized(self, since: datetime) -> List[Tuple]:
    """Extract impressions with optimized query performance."""
    query = """
        SELECT id, campaign_id, created_at
        FROM impressions
        WHERE created_at > %s
        ORDER BY created_at
        LIMIT 50000  -- Batch size control
    """
    return self.db.execute_query(query, (since,))
```

Additionally, I recommend adding appropriate indexes to source tables:

```sql
-- Add index for incremental extraction performance
CREATE INDEX IF NOT EXISTS idx_impressions_created_at ON impressions(created_at);
CREATE INDEX IF NOT EXISTS idx_clicks_created_at ON clicks(created_at);
```

### 2. Batch Processing and Bulk Loading

Implementing bulk loading operations for ClickHouse significantly improves throughput:

```python
def load_impressions_bulk(self, data: List[Tuple]) -> int:
    """Load impression data into ClickHouse using efficient bulk loading."""
    if not data:
        return 0
    
    # Prepare data for bulk insert
    formatted_data = []
    for imp_id, campaign_id, event_date, event_time, created_at in data:
        formatted_data.append({
            'impression_id': imp_id,
            'campaign_id': campaign_id,
            'event_date': event_date,
            'event_time': event_time,
            'created_at': created_at
        })
    
    # Execute bulk insert
    self.db.client.execute(
        "INSERT INTO analytics.fact_impressions VALUES",
        formatted_data
    )
    return len(data)
```

### 3. Memory Management

For large datasets, implement iterator-based processing to avoid loading entire result sets into memory:

```python
def extract_large_dataset(self, since: datetime, batch_size=10000):
    """Extract large datasets using server-side cursors to control memory usage."""
    query = """
        SELECT id, campaign_id, created_at 
        FROM impressions
        WHERE created_at > %s
        ORDER BY id
    """
    
    with self.db.conn.cursor(name='large_extract_cursor') as cursor:
        cursor.execute(query, (since,))
        
        while True:
            batch = cursor.fetchmany(batch_size)
            if not batch:
                break
            yield batch
```

### 4. Compression and Data Type Optimizations

Optimize ClickHouse storage by carefully selecting compression and data types:

```sql
-- Optimized fact table with compression and efficient data types
CREATE TABLE IF NOT EXISTS analytics.fact_impressions_optimized
(
    impression_id UInt32 CODEC(Delta, LZ4),
    campaign_id UInt32 CODEC(Delta, LZ4),
    event_date Date CODEC(DoubleDelta, LZ4),
    event_time DateTime CODEC(DoubleDelta, LZ4),
    created_at DateTime CODEC(DoubleDelta, LZ4)
) ENGINE = MergeTree()
PARTITION BY toYYYYMM(event_date)
ORDER BY (campaign_id, event_date)
SETTINGS index_granularity = 8192;
```

### 5. Parallel Processing for Transformations

Implement parallel transformation processing for CPU-intensive operations:

```python
def transform_impressions_parallel(self, rows: List[Tuple]) -> List[Tuple]:
    """Transform impression data using parallel processing."""
    from concurrent.futures import ProcessPoolExecutor
    
    # Split data into chunks for parallel processing
    chunk_size = 10000
    chunks = [rows[i:i + chunk_size] for i in range(0, len(rows), chunk_size)]
    
    # Process chunks in parallel
    with ProcessPoolExecutor() as executor:
        results = list(executor.map(self._transform_impression_chunk, chunks))
    
    # Combine results
    return [item for sublist in results for item in sublist]
```

## Future-Proofing the Architecture

As digital advertising continues to evolve, this ETL pipeline can be extended in several ways:

### 1. Streaming Data Processing

Implement real-time data processing by integrating with streaming platforms:

```python
class StreamingETLPipeline:
    """Real-time streaming ETL pipeline for advertising events."""
    
    def __init__(self, kafka_config, clickhouse_connector):
        self.consumer = KafkaConsumer(
            'adtech.events',
            bootstrap_servers=kafka_config['bootstrap_servers'],
            group_id='etl-consumer-group',
            auto_offset_reset='earliest'
        )
        self.clickhouse = clickhouse_connector
        
    def process_streaming_events(self):
        """Process streaming events from Kafka."""
        for message in self.consumer:
            event = json.loads(message.value.decode('utf-8'))
            
            # Process different event types
            if event['type'] == 'impression':
                self.process_impression(event)
            elif event['type'] == 'click':
                self.process_click(event)
                
    def process_impression(self, event):
        """Process impression event and load to ClickHouse."""
        # Transform and load impression data
        self.clickhouse.execute_query(
            "INSERT INTO analytics.fact_impressions VALUES",
            [(
                event['id'],
                event['campaign_id'],
                datetime.fromisoformat(event['timestamp']).date(),
                datetime.fromisoformat(event['timestamp']),
                datetime.now()
            )]
        )
```

### 2. Machine Learning Feature Store Integration

Extend the pipeline to support ML feature generation for predictive advertising:

```python
class FeatureStoreLoader:
    """Loads transformed data into a feature store for ML applications."""
    
    def __init__(self, clickhouse_connector, feature_store_client):
        self.clickhouse = clickhouse_connector
        self.feature_store = feature_store_client
        
    def generate_campaign_features(self):
        """Generate and load campaign performance features."""
        # Extract features from ClickHouse
        features_data = self.clickhouse.execute_query("""
            SELECT 
                campaign_id,
                toDate(event_time) AS day,
                count() AS daily_impressions,
                sum(case when exists(
                    SELECT 1 FROM analytics.fact_clicks c 
                    WHERE c.campaign_id = i.campaign_id AND c.event_date = i.event_date
                ) then 1 else 0 end) AS daily_clicks,
                avg(bid) AS avg_bid
            FROM analytics.fact_impressions i
            JOIN analytics.dim_campaign c ON i.campaign_id = c.campaign_id
            GROUP BY campaign_id, day
            ORDER BY campaign_id, day
        """)
        
        # Load to feature store
        self.feature_store.ingest_features(
            feature_group="campaign_daily_performance",
            features=features_data
        )
```

### 3. Multi-Tenant Architecture

Scale the system to support multiple advertising platforms through tenant isolation:

```python
class MultiTenantETLPipeline(ETLPipeline):
    """ETL pipeline with tenant isolation support."""
    
    def __init__(self, tenant_id, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tenant_id = tenant_id
        
    def extract_tenant_data(self, table, since):
        """Extract data specifically for this tenant."""
        query = f"""
            SELECT * FROM {table} 
            WHERE tenant_id = %s AND updated_at > %s
        """
        return self.extractor.db.execute_query(query, (self.tenant_id, since))
        
    def load_tenant_data(self, table, data):
        """Load data with tenant isolation."""
        # Ensure tenant_id is included in all loaded data
        tenant_data = [(self.tenant_id,) + row for row in data]
        return self.loader.load_generic(table, tenant_data)
```

## Conclusion

Building this ETL pipeline for AdTech analytics was a structured exercise in balancing performance, reliability, and maintainability. The modular design allows for easy extension to additional data sources or targets, while the ClickHouse optimizations ensure fast query responses for critical KPIs.

While the current implementation meets the immediate needs, I've outlined clear paths for scaling and optimizing as data volumes grow. The real power of this architecture lies in its flexibility—it can evolve from batch processing to streaming, accommodate multi-tenant requirements, and integrate with advanced analytics platforms as business needs change.

For organizations starting similar projects, I recommend taking an incremental approach: begin with the core batch ETL functionality, validate the data model with real queries, then progressively enhance with optimizations and scalability features based on actual performance metrics and growth patterns.

This architectural pattern has proven highly effective for advertising analytics, where the ability to process billions of events while maintaining query performance is critical to driving business value through data-driven decision making.
