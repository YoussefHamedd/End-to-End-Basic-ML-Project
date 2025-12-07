# Monitoring Guide - Prometheus + Grafana

Complete guide for monitoring your Fake News Detection ML model in production.

---

## 🎯 What's Monitored

### Application Metrics
- **Total Predictions**: Count of all predictions made
- **Prediction Rate**: Predictions per second
- **Prediction Latency**: Time taken for each prediction (p50, p95)
- **Model Confidence**: Latest prediction confidence score
- **Predictions by Type**: Count of Real News vs Fake News predictions
- **API Request Status**: Success/Error counts
- **Error Count**: Total errors by type

### System Metrics
- **Prometheus**: Self-monitoring metrics
- **MLflow**: (Optional) Experiment tracking metrics

---

## ⚡ Quick Start (5 minutes)

### Option 1: Full Stack with Docker Compose (Recommended)

```bash
# Start everything (API + MLflow + Prometheus + Grafana)
docker-compose up -d

# Wait 30 seconds for services to start
sleep 30

# Access services:
# - API: http://localhost:8080
# - MLflow: http://localhost:5000
# - Prometheus: http://localhost:9090
# - Grafana: http://localhost:3000 (admin/admin)
```

### Option 2: Local Development

```bash
# Terminal 1: Start Flask API
python app.py

# Terminal 2: Start Prometheus (requires Docker)
docker run -d \
  --name prometheus \
  -p 9090:9090 \
  -v $(pwd)/monitoring/prometheus.yml:/etc/prometheus/prometheus.yml \
  prom/prometheus

# Terminal 3: Start Grafana (requires Docker)
docker run -d \
  --name=grafana \
  -p 3000:3000 \
  grafana/grafana
```

---

## 📊 Accessing Dashboards

### Grafana Dashboard

1. **Open Grafana**: http://localhost:3000
2. **Login**:
   - Username: `admin`
   - Password: `admin`
3. **Navigate to Dashboard**:
   - Go to Dashboards → "Fake News Detection - ML Model Monitoring"
   - Or search for "Fake News"

### Dashboard Panels

1. **Total Predictions** - Stat panel showing total predictions
2. **Prediction Rate** - Time series of predictions per second
3. **Prediction Latency** - p50 and p95 latency over time
4. **Current Model Confidence** - Gauge showing latest confidence
5. **Total Errors** - Error count stat
6. **Predictions by Type** - Bar chart (Real vs Fake)
7. **API Request Status** - Success/Error trends

### Prometheus UI

1. **Open Prometheus**: http://localhost:9090
2. **Query Metrics**:
   - Navigate to Graph
   - Enter metric name (e.g., `fakenews_predictions_total`)
   - Click Execute

**Useful Queries**:

```promql
# Prediction rate (per second)
rate(fakenews_predictions_total[5m])

# 95th percentile latency
histogram_quantile(0.95, rate(fakenews_prediction_duration_seconds_bucket[5m]))

# Error rate
rate(fakenews_errors_total[5m])

# Predictions by type
sum by(prediction_type) (fakenews_predictions_total)

# API success rate
sum(fakenews_api_requests_total{status="success"}) / sum(fakenews_api_requests_total)
```

---

## 🔍 Available Metrics

### Counters
| Metric | Description | Labels |
|--------|-------------|--------|
| `fakenews_predictions_total` | Total predictions made | `prediction_type` |
| `fakenews_api_requests_total` | Total API requests | `endpoint`, `method`, `status` |
| `fakenews_errors_total` | Total errors | `error_type` |

### Histograms
| Metric | Description |
|--------|-------------|
| `fakenews_prediction_duration_seconds` | Prediction processing time |

### Gauges
| Metric | Description |
|--------|-------------|
| `fakenews_model_confidence` | Latest model confidence score |

---

## 🧪 Testing Metrics

### Generate Test Predictions

```bash
# Single prediction
curl -X POST http://localhost:8080/predict \
  -H "Content-Type: application/json" \
  -d '{
    "title": "Breaking News",
    "text": "Scientists discover breakthrough in renewable energy technology."
  }'

# Batch predictions
curl -X POST http://localhost:8080/predict_batch \
  -H "Content-Type: application/json" \
  -d '{
    "articles": [
      {"title": "News 1", "text": "Article text 1..."},
      {"title": "News 2", "text": "Article text 2..."}
    ]
  }'
```

### View Raw Metrics

```bash
# Get Prometheus-formatted metrics
curl http://localhost:8080/metrics

# Example output:
# # HELP fakenews_predictions_total Total predictions made
# # TYPE fakenews_predictions_total counter
# fakenews_predictions_total{prediction_type="Real News"} 42.0
# fakenews_predictions_total{prediction_type="Fake News"} 28.0
```

---

## 🚨 Setting Up Alerts

### Prometheus Alert Rules

Create `monitoring/alerts.yml`:

```yaml
groups:
  - name: fakenews_alerts
    interval: 30s
    rules:
      # High error rate
      - alert: HighErrorRate
        expr: rate(fakenews_errors_total[5m]) > 0.1
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High error rate detected"
          description: "Error rate is {{ $value }} errors/sec"

      # High latency
      - alert: HighLatency
        expr: histogram_quantile(0.95, rate(fakenews_prediction_duration_seconds_bucket[5m])) > 2
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High prediction latency"
          description: "95th percentile latency is {{ $value }}s"

      # Low confidence predictions
      - alert: LowConfidence
        expr: fakenews_model_confidence < 0.6
        for: 1m
        labels:
          severity: info
        annotations:
          summary: "Low model confidence"
          description: "Model confidence is {{ $value }}"

      # API down
      - alert: APIDown
        expr: up{job="flask-api"} == 0
        for: 2m
        labels:
          severity: critical
        annotations:
          summary: "API is down"
          description: "Flask API has been down for 2 minutes"
```

Update `monitoring/prometheus.yml`:

```yaml
global:
  scrape_interval: 15s

# Add alert rules
rule_files:
  - "alerts.yml"

# Add Alertmanager (optional)
alerting:
  alertmanagers:
    - static_configs:
        - targets: ['alertmanager:9093']

scrape_configs:
  - job_name: 'flask-api'
    static_configs:
      - targets: ['ml-app:8080']
    metrics_path: '/metrics'
```

---

## 📈 Grafana Advanced Configuration

### Create Custom Dashboards

1. In Grafana, click "+" → "Dashboard"
2. Add Panel
3. Select Prometheus datasource
4. Enter query (e.g., `rate(fakenews_predictions_total[5m])`)
5. Customize visualization
6. Save dashboard

### Import Dashboard

```bash
# Copy pre-built dashboard
cp monitoring/grafana/dashboards/fakenews_dashboard.json /path/to/import/

# In Grafana:
# 1. Click "+" → "Import"
# 2. Upload JSON file or paste JSON
# 3. Select Prometheus datasource
# 4. Click "Import"
```

### Variables & Templating

Add variables for dynamic filtering:

```
Variable Name: environment
Type: Query
Query: label_values(fakenews_predictions_total, environment)
```

Use in queries:
```promql
rate(fakenews_predictions_total{environment="$environment"}[5m])
```

---

## 🐳 Docker Compose Configuration

### Full Stack

```yaml
services:
  ml-app:
    build: .
    ports:
      - "8080:8080"
    depends_on:
      - mlflow

  mlflow:
    image: ghcr.io/mlflow/mlflow:latest
    ports:
      - "5000:5000"

  prometheus:
    image: prom/prometheus:latest
    ports:
      - "9090:9090"
    volumes:
      - ./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml

  grafana:
    image: grafana/grafana:latest
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
    volumes:
      - ./monitoring/grafana/dashboards:/etc/grafana/provisioning/dashboards
      - ./monitoring/grafana/datasources:/etc/grafana/provisioning/datasources
```

---

## 🔧 Troubleshooting

### Metrics Not Showing in Prometheus

1. **Check API metrics endpoint**:
   ```bash
   curl http://localhost:8080/metrics
   ```

2. **Check Prometheus targets**:
   - Open http://localhost:9090/targets
   - Ensure `flask-api` target is "UP"

3. **Check Prometheus config**:
   ```bash
   docker exec prometheus cat /etc/prometheus/prometheus.yml
   ```

### Grafana Shows No Data

1. **Verify datasource**:
   - Go to Configuration → Data Sources
   - Click Prometheus → Test
   - Should show "Data source is working"

2. **Check time range**:
   - Ensure time range includes recent data
   - Try "Last 15 minutes"

3. **Verify metrics exist**:
   - Go to Explore
   - Select Prometheus
   - Run query: `fakenews_predictions_total`

### Docker Compose Issues

```bash
# Check service logs
docker-compose logs ml-app
docker-compose logs prometheus
docker-compose logs grafana

# Restart services
docker-compose restart

# Full reset
docker-compose down -v
docker-compose up -d
```

---

## 📊 Best Practices

### 1. Metric Naming
- Use descriptive names: `fakenews_predictions_total`
- Follow Prometheus conventions: `<namespace>_<metric>_<unit>`
- Use labels for dimensions: `{prediction_type="Real News"}`

### 2. Alerting
- Alert on symptoms, not causes
- Set appropriate thresholds based on baseline
- Include runbook links in alert annotations
- Use alert severity levels (critical, warning, info)

### 3. Dashboard Design
- Group related metrics
- Use appropriate visualization types
- Add descriptions to panels
- Set reasonable time ranges

### 4. Data Retention
- Configure Prometheus retention:
  ```yaml
  storage:
    tsdb:
      retention.time: 15d
      retention.size: 10GB
  ```

### 5. Performance
- Use recording rules for expensive queries
- Limit cardinality of labels
- Use appropriate scrape intervals

---

## 📖 Additional Resources

- **Prometheus Documentation**: https://prometheus.io/docs/
- **Grafana Documentation**: https://grafana.com/docs/
- **PromQL Tutorial**: https://prometheus.io/docs/prometheus/latest/querying/basics/
- **Grafana Dashboard Gallery**: https://grafana.com/grafana/dashboards/

---

## ✅ Week 7 Checklist (Academic Project)

- [x] Prometheus metrics integrated into Flask API
- [x] Grafana dashboard created
- [x] Docker Compose configured
- [x] Metrics endpoint (`/metrics`) working
- [ ] Alert rules configured
- [ ] Data retention policy set
- [ ] Monitoring documentation complete

---

## 🎓 For Your Report

**Monitoring Implementation**:

1. **Metrics Collected**:
   - Prediction volume, latency, confidence
   - API request rates and error rates
   - System health metrics

2. **Tools Used**:
   - Prometheus for metrics collection
   - Grafana for visualization
   - Docker Compose for orchestration

3. **Key Insights**:
   - Real-time model performance monitoring
   - Early detection of anomalies
   - Production readiness validation

4. **Screenshots to Include**:
   - Grafana dashboard with live metrics
   - Prometheus targets page
   - Alert configuration
   - Sample metrics output

---

**Happy Monitoring!** 🚀
