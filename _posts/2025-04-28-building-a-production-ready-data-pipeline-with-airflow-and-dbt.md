---
layout: post
title: 'Building a Production-Ready Data Pipeline with Airflow and dbt'
description: A comprehensive walkthrough of implementing a modern sales data engineering pipeline with Airflow, dbt, and PostgreSQL, focusing on practical insights and real-world patterns.
image: 
category: [Data Engineering]
tags: python postgres airflow dbt etl
date: 2025-04-28 14:30 +0200
---

In today's data-driven business landscape, transforming raw operational data into actionable insights requires robust, scalable data pipelines. I recently designed and implemented a comprehensive data engineering solution for sales analytics that bridges raw transactional data to dimensional models. This post walks through the architecture, implementation decisions, and practical patterns that you can apply to your own data engineering projects.

## The Challenge: From Sales Transactions to Analytics

The core challenge was to build a pipeline that would:

1. Ingest sales transaction data from multiple CSV sources
2. Handle common data quality issues (missing values, inconsistent formats)
3. Transform raw data into a dimensional model for analytics
4. Implement incremental processing for efficiency
5. Ensure reliability with comprehensive testing and error handling

## Architecture Design: Medallion Pattern Implementation

After evaluating several architectural approaches, I implemented a medallion architecture (also called multi-hop architecture), which organizes data through progressive refinement stages:

```
Raw CSV Data → Bronze Layer (raw.sales) → Silver Layer (transformed.dim_*) → Gold Layer (analytics.fact_sales)
```

This layered approach provides several key advantages:
- Clear separation of concerns
- Progressive data quality improvement
- Complete data lineage traceability
- Flexibility to rebuild downstream layers without re-ingesting source data

### Database Schema Design

I designed the following schema structure:

- **Bronze Layer**: Raw data storage with original values preserved
  - `raw.sales`: Original CSV data with added metadata columns

- **Silver Layer**: Cleaned and transformed dimensional models
  - `transformed.dim_product`: Product information
  - `transformed.dim_retailer`: Retailer information
  - `transformed.dim_location`: Location information
  - `transformed.dim_channel`: Sales channel information
  - `transformed.dim_date`: Date dimension with hierarchies
  - `transformed.fact_sales`: Sales fact table with foreign keys and measures

- **Gold Layer**: Analytics-ready views and aggregates
  - `analytics.dim_*`: Analytics-ready dimension views
  - `analytics.fact_sales`: Optimized analytical fact table

## Implementation: Core Components

### 1. Data Ingestion Module

The ingestion component follows a "validate-first, then clean" pattern, which provides better visibility into data quality issues:

```python
def main(file_path):
    """
    Main function to process and load data.
    New flow: Validate first, then clean only records that need cleaning.
    """
    logger.info(f"Starting data ingestion process for {file_path}")
    
    try:
        # Read the CSV file
        logger.info(f"Reading file: {file_path}")
        df = pd.read_csv(file_path)
        logger.info(f"Successfully read {len(df)} records from {file_path}")
        
        # First validate the data as-is
        is_valid, invalid_indices = validate_data(df)
        
        if not is_valid:
            logger.info("Data validation failed. Applying cleaning steps...")
            # Apply cleaning only after validation fails
            df = CleanData.apply_all_cleaners(df)
            
            # Re-validate after cleaning
            is_valid, invalid_indices = validate_data(df)
            if not is_valid:
                logger.warning("Data still contains invalid records after cleaning. Filtering them out.")
                df = filter_invalid_records(df, invalid_indices)
            else:
                logger.info("Data cleaning resolved all validation issues.")
        else:
            logger.info("Data passed validation without cleaning.")
        
        # Check for duplicates
        deduplicated_df = detect_duplicates(df)
        
        # Only proceed with loading if we have records
        if len(deduplicated_df) > 0:
            # Get database connection string
            connection_string = get_db_connection_string()
            
            # Load to raw schema
            records_loaded = load_to_raw(deduplicated_df, connection_string, file_path)
            
            logger.info(f"Data ingestion complete. {records_loaded} records processed.")
            return records_loaded
        else:
            logger.warning("No valid records to load after validation and deduplication.")
            return 0
        
    except Exception as e:
        logger.error(f"Error in data ingestion process: {str(e)}")
        raise
```

I encapsulated the cleaning operations in a dedicated class with specialized methods for each type of cleaning:

```python
class CleanData:
    """
    A class for handling different types of data cleaning operations.
    Each method handles a specific type of cleaning.
    """
    
    @staticmethod
    def handle_missing_values(df):
        """Handle missing values in the DataFrame."""
        logger.info("Cleaning: Handling missing values...")
        df_cleaned = df.copy()
        df_cleaned['Location'] = df_cleaned['Location'].fillna('Unknown')
        return df_cleaned
    
    @staticmethod
    def clean_price_values(df):
        """Clean price values by removing currency symbols."""
        logger.info("Cleaning: Cleaning price values...")
        df_cleaned = df.copy()
        
        # Handle price with currency notation
        df_cleaned.loc[df_cleaned['Price'].str.contains('USD', na=False), 'Price'] = \
            df_cleaned.loc[df_cleaned['Price'].str.contains('USD', na=False), 'Price'].str.replace('USD', '')
        
        # Strip whitespace
        df_cleaned['Price'] = df_cleaned['Price'].str.strip()
        
        return df_cleaned
    
    # More cleaning methods...
    
    @classmethod
    def apply_all_cleaners(cls, df):
        """Apply all cleaning methods in sequence."""
        logger.info("Starting comprehensive data cleaning...")
        
        df_result = df.copy()
        df_result = cls.handle_missing_values(df_result)
        df_result = cls.standardize_data_types(df_result)
        df_result = cls.remove_whitespace_values(df_result)
        df_result = cls.clean_price_values(df_result)
        df_result = cls.clean_date_values(df_result)
        
        logger.info(f"Comprehensive data cleaning complete. Processed {len(df_result)} rows.")
        return df_result
```

### 2. Data Transformation Layer

After landing raw data, the transformation component converts it into a proper dimensional model:

```python
def process_sales_data(engine):
    """
    Process and transform sales data from raw to fact table.
    """
    
    try:
        # Get dimension lookups
        channel_ids = populate_dim_channel(engine)
        location_ids = populate_dim_location(engine)
        product_ids = populate_dim_product(engine)
        retailer_ids = populate_dim_retailer(engine)
        date_ids = populate_dim_date(engine)
        
        # Query to get raw sales data
        query = """
        SELECT "SaleID", "ProductID", "RetailerID", "Channel", "Location", 
               "Quantity", "Price", "Date"
        FROM raw.sales
        WHERE "SaleID" NOT IN (
            SELECT sale_id::VARCHAR FROM transformed.fact_sales
        )
        ORDER BY "SaleID" ASC
        """
        
        with engine.begin() as conn:
            # Count total records to process
            count_query = """
            SELECT COUNT(*) FROM raw.sales
            WHERE "SaleID" NOT IN (
                SELECT sale_id::VARCHAR FROM transformed.fact_sales
            )
            """
            result = conn.execute(text(count_query))
            total_records = result.fetchone()[0]
            logger.info(f"Found {total_records} new sales records to process")
            
            if total_records == 0:
                logger.info("No new records to process")
                return 0
            
            result = conn.execute(text(query))
            sales = [dict(zip(result.keys(), row)) for row in result]
            
            # Transform data
            logger.info(f"Processing {len(sales)} sales records")
            processed_count = 0
            fact_records = []
            
            # Process each sale record
            for sale in sales:
                try:
                    # Data transformations here...
                    
                    # Create fact record
                    fact_record = {
                        "sale_id": sale_id,
                        "product_id": product_id,
                        "retailer_id": retailer_id,
                        "location_id": location_id,
                        "channel_id": channel_id,
                        "date_id": date_id,
                        "quantity": quantity,
                        "unit_price": unit_price,
                        "total_amount": total_amount
                    }
                    fact_records.append(fact_record)
                except Exception as e:
                    logger.error(f"Error processing sale {sale['SaleID']}: {str(e)}")
            
            # Insert fact records
            if fact_records:
                try:
                    insert_query = """
                    INSERT INTO transformed.fact_sales (
                        sale_id, product_id, retailer_id, location_id, 
                        channel_id, date_id, quantity, unit_price, total_amount
                    )
                    VALUES (
                        :sale_id, :product_id, :retailer_id, :location_id, 
                        :channel_id, :date_id, :quantity, :unit_price, :total_amount
                    )
                    ON CONFLICT (sale_id) DO NOTHING
                    """
                    # Use a new transaction to ensure atomicity
                    with engine.begin() as insert_conn:
                        insert_conn.execute(text(insert_query), fact_records)
                    
                    processed_count = len(fact_records)
                    logger.info(f"Inserted {len(fact_records)} records into fact_sales")
                except Exception as e:
                    logger.error(f"Error inserting: {str(e)}")
            
            logger.info(f"Successfully processed {processed_count} sales records")
            return processed_count
    except Exception as e:
        logger.error(f"Error processing sales data: {str(e)}")
        raise
```

### 3. dbt Transformation Models

For the analytical layer, I implemented dbt models that further refine the data:

First, a staging model to standardize raw data formats:

```sql
-- stg_sales.sql
{% raw %}
with source as (
    select * from {{ source('postgres', 'sales') }}
),{% endraw %}

cleaned as (
    select
        "SaleID"::integer as sale_id,
        nullif("ProductID", '')::integer as product_id,
        "ProductName" as product_name,
        "Brand" as brand,
        "Category" as category,
        "RetailerID"::integer as retailer_id,
        "RetailerName" as retailer_name,
        "Channel" as channel,
        coalesce(nullif("Location", ''), 'Unknown') as location,
        case 
            when "Quantity" ~ '^-?\d+$' then "Quantity"::integer
            else null
        end as quantity,
        case
            when "Price" ~ '^\d+$' then "Price"::decimal
            when "Price" ~ '^\d+USD$' then replace("Price", 'USD', '')::decimal
            else null
        end as price,
        case
            when "Date" ~ '^\d{4}-\d{2}-\d{2}$' then "Date"::date
            when "Date" ~ '^\d{4}/\d{2}/\d{2}$' then to_date("Date", 'YYYY/MM/DD')
            else null
        end as date,
        batch_id,
        source_file,
        inserted_at
    from source
),

final as (
    select
        sale_id,
        product_id,
        product_name,
        brand,
        category,
        retailer_id,
        retailer_name,
        channel,
        location,
        case when quantity <= 0 then null else quantity end as quantity,
        price,
        date,
        batch_id,
        source_file,
        inserted_at,
        current_timestamp as transformed_at
    from cleaned
    where 
        sale_id is not null
        and product_id is not null
        and retailer_id is not null
        and date is not null
        and quantity is not null
        and price is not null
)

select * from final
```

Then dimensional models built on top of staging:

```sql
-- fact_sales.sql
{% raw %}
{{
  config(
    unique_key = 'sale_id',
    indexes = [
      {'columns': ['sale_id'], 'unique': True},
      {'columns': ['product_id']},
      {'columns': ['retailer_id']},
      {'columns': ['location_id']},
      {'columns': ['channel_id']},
      {'columns': ['date_id']}
    ]
  )
}}

with stg_sales as (
    select * from {{ ref('stg_sales') }}
),

dim_product as (
    select * from {{ ref('dim_product') }}
),

dim_location as (
    select * from {{ ref('dim_location') }}
),

-- Create dimension references for retailer and channel
dim_retailer as (
    select distinct
        retailer_id,
        retailer_name
    from stg_sales
),

dim_channel as (
    select
        channel,
        {{ dbt_utils.generate_surrogate_key(['channel']) }} as channel_id
    from stg_sales
    group by channel
),

-- Final fact sales table
final as (
    select
        s.sale_id,
        s.product_id,
        s.retailer_id,
        l.location_id,
        c.channel_id,
        s.date as date_id,
        s.quantity,
        s.price / nullif(s.quantity, 0)::numeric(10, 2) as unit_price,
        s.price::numeric(12, 2) as total_amount,
        s.transformed_at
    from stg_sales s
    inner join dim_location l on l.location = s.location
    inner join dim_channel c on c.channel = s.channel
    {% if is_incremental() %}
    where s.transformed_at > (select max(transformed_at) from {{ this }})
    {% endif %}
)

select * from final
{% endraw %}
```

### 4. Orchestration with Airflow

The Airflow DAGs orchestrate the entire pipeline. I implemented two primary DAGs:

1. The Sales Data Pipeline for ingestion and initial transformation:

```python
# Define task dependencies
check_file_exists >> check_and_ingest_data >> transform_raw_data >> archive_file
```

2. The dbt Transformation Pipeline for analytical models:

```python
# Define dbt commands
dbt_deps_cmd = f"""
cd {DBT_PROJECT_DIR} && 
dbt deps --profiles-dir {DBT_PROFILES_DIR} --target {DBT_TARGET}
"""

dbt_run_staging_cmd = f"""
cd {DBT_PROJECT_DIR} && 
dbt run --models "staging.*" --profiles-dir {DBT_PROFILES_DIR} --target {DBT_TARGET}
"""

dbt_run_marts_cmd = f"""
cd {DBT_PROJECT_DIR} && 
dbt run --models "marts.*" --profiles-dir {DBT_PROFILES_DIR} --target {DBT_TARGET}
"""

dbt_test_cmd = f"""
cd {DBT_PROJECT_DIR} && 
dbt test --profiles-dir {DBT_PROFILES_DIR} --target {DBT_TARGET}
"""

# Define task dependencies
check_and_ingest_data >> install_dependencies >> run_staging_models >> run_mart_models >> test_models
```

## Advanced Features

### Self-healing Data Flow

A key feature of this pipeline is its "self-healing" capability. Both DAGs automatically check if the required data exists before proceeding, and trigger upstream processes if needed:

```python
class RawDataSensor(BaseSensorOperator):
    """
    Sensor to check if there's data in the raw.sales table.
    """
    @apply_defaults
    def __init__(self, conn_id="sales_db", *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.conn_id = conn_id

    def poke(self, context):
        hook = PostgresHook(postgres_conn_id=self.conn_id)
        sql = "SELECT COUNT(*) FROM raw.sales"
        count = hook.get_first(sql)[0]
        self.log.info(f"Found {count} rows in raw.sales table")
        return count > 0


def check_and_ingest_data(csv_file_path, conn_id="sales_db", **context):
    """
    Check if data exists in raw.sales and ingest if empty.
    """
    hook = PostgresHook(postgres_conn_id=conn_id)
    
    # Check if data exists
    sql = "SELECT COUNT(*) FROM raw.sales"
    count = hook.get_first(sql)[0]
    
    # If data exists, return True
    if count > 0:
        context['ti'].xcom_push(key='data_already_exists', value=True)
        return True
    
    # If no data, perform ingestion
    try:
        ingest_main = import_ingest_module()
        records_loaded = ingest_main(csv_file_path)
        
        # Verify ingestion was successful
        if records_loaded > 0:
            context['ti'].xcom_push(key='records_loaded', value=records_loaded)
            return True
        else:
            context['ti'].xcom_push(key='ingest_failed', value=True)
            return False
    except Exception as e:
        context['ti'].xcom_push(key='ingest_error', value=str(e))
        raise
```

This design enables either pipeline to be triggered independently without failures, creating a more resilient system.

### Data Quality Testing

Comprehensive data quality checks are implemented in both Python and dbt:

1. Python validation for source data:

```python
def validate_data(df):
    """
    Validate data quality and identify invalid records.
    Returns a boolean indicating if the data is valid and a list of invalid indices.
    """
    logger.info(f"Validating data... Total records: {len(df)}")
    
    # Track invalid rows for logging
    invalid_rows = {
        'dates': [],
        'quantities': [],
        'prices': [],
        'all': set()  # Use a set to avoid duplicates
    }
    
    # Check for invalid dates
    for idx, date_str in enumerate(df['Date']):
        try:
            # Try to parse the date
            if isinstance(date_str, str) and date_str:
                # Handle different date formats
                if '/' in date_str:
                    datetime.strptime(date_str, '%Y/%m/%d')
                else:
                    datetime.strptime(date_str, '%Y-%m-%d')
            else:
                # Empty or non-string date
                invalid_rows['dates'].append((idx, date_str))
                invalid_rows['all'].add(idx)
        except ValueError:
            invalid_rows['dates'].append((idx, date_str))
            invalid_rows['all'].add(idx)
    
    # More validation checks...
    
    return is_valid, list(invalid_rows['all'])
```

2. dbt tests for transformed data:

```yaml
# schema.yml for fact_sales
version: 2

models:
  - name: fact_sales
    description: "Fact table for sales with related dimension keys"
    columns:
      - name: sale_id
        description: "The primary key for the sales transaction"
        tests:
          - unique
          - not_null
      
      - name: product_id
        description: "Foreign key to product dimension"
        tests:
          - not_null
          - relationships:
              to: ref('dim_product')
              field: product_id
      
      # Additional column tests...
      
      - name: quantity
        description: "Number of items sold"
        tests:
          - not_null
          - positive_values
```

### Incremental Processing

The pipeline implements incremental processing at multiple levels:

1. Python-based ETL uses ID-based tracking:

```python
# Query to get only new records 
query = """
SELECT "SaleID", "ProductID", "RetailerID", "Channel", "Location", 
       "Quantity", "Price", "Date"
FROM raw.sales
WHERE "SaleID" NOT IN (
    SELECT sale_id::VARCHAR FROM transformed.fact_sales
)
ORDER BY "SaleID" ASC
"""
```

2. dbt models use incremental materialization:

```sql
{% raw %}
{% if is_incremental() %}
where s.transformed_at > (select max(transformed_at) from {{ this }})
{% endif %}
{% endraw %}
```

This two-tiered approach ensures efficient processing of only new or changed data.

## Containerization and Deployment

The entire solution is containerized using Docker for consistent deployment:

```yaml
services:
  postgres:
    image: postgres:latest
    environment:
      - POSTGRES_USER=postgres
      - POSTGRES_PASSWORD=mysecretpassword
      - POSTGRES_MULTIPLE_DATABASES=airflow,sales
    volumes:
      - ./initdb:/docker-entrypoint-initdb.d
      - postgres-db-volume:/var/lib/postgresql/data
    healthcheck:
      test: ["CMD", "pg_isready", "-U", "postgres"]
      interval: 5s
      retries: 5
    ports:
      - "5433:5432"
    restart: always

  airflow-webserver:
    <<: *airflow-common
    command: webserver
    ports:
      - 8081:8080
    healthcheck:
      test: ["CMD", "curl", "--fail", "http://localhost:8081/health"]
      interval: 10s
      timeout: 10s
      retries: 5
    restart: always

  # Additional services...
```

## Scalability Approaches

As the data volume grows, several scalability enhancements can be implemented:

### 1. Partitioning for Larger Datasets

For larger datasets, implementing table partitioning in PostgreSQL can significantly improve performance:

```sql
-- Example of adding partitioning to fact_sales
CREATE TABLE analytics.fact_sales (
    sale_id INTEGER,
    -- other columns...
    date_id DATE NOT NULL
) PARTITION BY RANGE (date_id);

-- Create partitions by month
CREATE TABLE fact_sales_2024_q1 PARTITION OF fact_sales
    FOR VALUES FROM ('2024-01-01') TO ('2024-04-01');

CREATE TABLE fact_sales_2024_q2 PARTITION OF fact_sales
    FOR VALUES FROM ('2024-04-01') TO ('2024-07-01');
```

### 2. Parallel Processing with Airflow

For processing large volumes of data, implementing parallel task execution in Airflow:

```python
# Create parallel tasks for processing different data segments
for segment in get_data_segments():
    process_segment = PythonOperator(
        task_id=f'process_segment_{segment}',
        python_callable=process_data_segment,
        op_kwargs={'segment': segment},
        dag=dag,
    )
    
    # Set dependencies
    check_and_ingest_data >> process_segment >> merge_segments
```

### 3. Enhanced Incremental Processing

The current incremental approach can be enhanced with timestamp-based windowing:

```python
def extract_incremental_data(start_time, end_time, batch_size=10000):
    """Extract data in time-bounded batches for efficient processing."""
    current_position = start_time
    
    while current_position < end_time:
        next_position = min(current_position + timedelta(hours=1), end_time)
        
        query = """
        SELECT * FROM raw.sales
        WHERE inserted_at >= %s AND inserted_at < %s
        ORDER BY inserted_at
        LIMIT %s
        """
        
        yield execute_query(query, (current_position, next_position, batch_size))
        current_position = next_position
```

This approach reduces memory pressure when dealing with large datasets.

## Optimization Techniques

### 1. Database Indexing Strategy

Carefully designed indexes dramatically improve query performance:

```sql
-- Indexes for the fact table
CREATE INDEX idx_fact_sales_product_id ON transformed.fact_sales(product_id);
CREATE INDEX idx_fact_sales_retailer_id ON transformed.fact_sales(retailer_id);
CREATE INDEX idx_fact_sales_date_id ON transformed.fact_sales(date_id);
CREATE INDEX idx_fact_sales_channel_id ON transformed.fact_sales(channel_id);
```

### 2. Memory-Efficient Processing

For large datasets, implement batch processing to control memory usage:

```python
def process_large_dataset(file_path, batch_size=10000):
    """Process large CSV files in batches to control memory usage."""
    # Use pandas chunking for memory efficiency
    for chunk in pd.read_csv(file_path, chunksize=batch_size):
        # Validate and clean the chunk
        is_valid, invalid_indices = validate_data(chunk)
        if not is_valid:
            chunk = CleanData.apply_all_cleaners(chunk)
            chunk = filter_invalid_records(chunk, invalid_indices)
            
        # Process the cleaned chunk
        load_to_raw(chunk, get_db_connection_string(), file_path)
```

### 3. Airflow Task Configuration

Optimizing Airflow task configuration for better resource utilization:

```python
# Task configuration for better resource management
task = PythonOperator(
    task_id='transform_raw_data',
    python_callable=transform_main,
    executor_config={
        'cpu_millicores': 1000,
        'memory_mb': 2048,
    },
    dag=dag,
)
```

## Conclusion

This data pipeline demonstrates how modern tools and architectural patterns can create a robust, production-ready data infrastructure. By combining Airflow's orchestration capabilities with dbt's transformation power and a well-designed schema, we've built a system that can handle real-world data challenges while maintaining flexibility for future growth.

Key takeaways from this implementation:

1. The value of a layered architectural approach (medallion pattern)
2. The importance of separating validation from cleaning for better data quality management
3. The benefits of self-healing data flows that can recover from failures
4. How containerization provides environment consistency across development and production

While no data pipeline is ever truly "complete" (data requirements evolve continuously), this implementation provides a solid foundation that can adapt to changing business needs. The patterns and practices demonstrated here can help create more resilient, maintainable data systems for organizations of any size.

The complete code for this project is available on GitHub at [samuelTyh/airflow-dbt-sales-analytics](https://github.com/samuelTyh/airflow-dbt-sales-analytics).

## Future Work

Looking ahead, this pipeline could be enhanced with:

1. **Real-time streaming capabilities**: Integrating a streaming solution like Kafka for near real-time data processing
2. **Advanced data quality monitoring**: Adding automated data quality monitoring with Great Expectations or dbt expectations
3. **ML feature engineering**: Extending the pipeline to generate features for machine learning models
4. **Cloud-native deployment**: Adapting the architecture for cloud platforms with services like AWS Glue, Azure Data Factory, or Google Cloud Dataflow

These enhancements would further extend the capabilities of the pipeline while maintaining the core architectural principles that make it reliable and maintainable.

What data pipeline patterns have you found most effective in your work? Share your thoughts and questions in the comments below!
