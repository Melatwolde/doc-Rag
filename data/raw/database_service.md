# Database Connection Management and High Availability

## Connection Pooling Strategy
To sustain heavy concurrent workloads and prevent thread starvation, database connections are managed through an asynchronous pool. The pool capacity is configured by the key `database.pool_size` (default: 20 connections) in `configs/config.yaml`.

During intense traffic spikes, the pool may burst beyond baseline up to `database.max_overflow` (default: 10 auxiliary connections). If all available connections remain saturated past the checkout timeout, the driver triggers error `ERR_CONN_TIMEOUT` followed by `ERR_POOL_EXHAUSTED`.

```python
def get_db_connection(timeout: float = 5.0):
    """Acquire a checked-out connection from the global pool or raise timeout."""
    connection = pool.acquire(timeout=timeout)
    if connection is None:
        raise DatabaseTimeoutError("ERR_CONN_TIMEOUT", "Exhausted connection pool")
    return connection
```

## Read Replica Routing and Load Distribution
To alleviate primary database contention and prevent bottlenecks on heavy reporting queries, read-only traffic is dynamically routed to replica clusters.

Replicas can be passed via the CLI argument `--db-replica-url` or specified in the service discovery environment. When routing read queries, the engine uses round-robin health checks with exponential backoff on retry. The flag `--max-connections` caps total sockets per worker process.
