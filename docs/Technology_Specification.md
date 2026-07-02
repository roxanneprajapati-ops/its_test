# Technology Specification

## AI-Based Traffic Signal Optimization System for Auckland Intersections

---

## 1. System Requirements

| Requirement | Specification |
|---|---|
| Operating System | Windows 10/11, macOS, or Linux |
| Python Version | 3.10 or higher (developed/tested on 3.12) |
| Minimum RAM | 4 GB |
| Disk Space | ~200 MB (code + dataset + outputs) |
| Internet Access | Not required at runtime (fully offline after install) |

---

## 2. Technology Stack

| Layer | Technology | Version (tested) | Purpose |
|---|---|---|---|
| Language | Python | 3.12 | Core implementation language |
| Data Processing | Pandas | ≥ 2.0 | DataFrame operations, CSV I/O |
| Numerical Computing | NumPy | ≥ 1.24 | Random number generation, array math |
| Machine Learning | Scikit-learn | ≥ 1.3 | Random Forest, Linear Regression, metrics |
| Model Persistence | Joblib | ≥ 1.3 | Saving/loading trained models |
| REST API | Flask | ≥ 3.0 | HTTP/JSON communication layer |
| Input Validation | Pydantic | ≥ 2.0 | Schema validation for API requests |
| Visualisation (static) | Matplotlib | ≥ 3.7 | Confusion matrix, trend charts |
| Visualisation (interactive) | Streamlit | ≥ 1.30 | Dashboard application |
| Security | hashlib (stdlib) | — | SHA-256 file integrity hashing |
| Testing | Custom assert-based suite (pytest-compatible) | — | Unit tests across all modules |

---

## 3. Architecture Pattern

The system follows a **layered pipeline architecture** combined with
**object-oriented design patterns**:

```
┌─────────────────────────────────────────────────────────┐
│  Perception Layer   →  simulator.py (TrafficSimulationManager)
├─────────────────────────────────────────────────────────┤
│  Processing Layer   →  preprocessing.py (PreprocessingPipeline)
├─────────────────────────────────────────────────────────┤
│  Application Layer  →  ml_model.py (ModelTrainingOrchestrator)
│                        optimizer.py (SignalOptimizationEngine)
├─────────────────────────────────────────────────────────┤
│  Security Layer     →  security.py (cross-cutting concern)
├─────────────────────────────────────────────────────────┤
│  Network Layer      →  api.py (Flask REST API)
├─────────────────────────────────────────────────────────┤
│  Presentation Layer →  dashboard.py (Streamlit)
└─────────────────────────────────────────────────────────┘
```

Each layer depends only on the layer(s) below it and communicates
through well-defined Python class interfaces (not global state),
satisfying separation of concerns and supporting independent testing.

---

## 4. Design Patterns Used

| Pattern | Implementation |
|---|---|
| **Abstract Base Class** | `TrafficDataGenerator` defines `generate()` as an abstract method |
| **Template / Strategy** | `IntersectionSimulator` implements intersection-specific simulation logic behind a common interface |
| **Composition** | `TrafficSimulationManager` composes multiple `IntersectionSimulator` objects |
| **Facade** | `ModelTrainingOrchestrator` simplifies training three different models into one `run()` call |
| **Data Transfer Object (Dataclass)** | `TrafficRecord`, `SignalRecommendation` |
| **Pipeline** | `PreprocessingPipeline` chains `DataCleaner` → `FeatureEngineer` |

---

## 5. Data Flow

1. `TrafficSimulationManager.save()` → writes `traffic_dataset_raw.csv`
2. `FileIntegrityChecker.generate_hash_file()` → writes `traffic_dataset.sha256`
3. `PreprocessingPipeline.run()` → cleans + engineers features → `traffic_dataset_processed.csv`
4. `ModelTrainingOrchestrator.run()` → trains models → `metrics.json`, `*.joblib`
5. `api.py` loads the trained `.joblib` models and serves predictions over HTTP
6. `dashboard.py` reads the processed CSV + `metrics.json` and renders visualisations

---

## 6. Security Specification

| Control | Standard / Technique | Notes |
|---|---|---|
| Authentication | Static API key via `X-API-Key` header | Demonstrates access control; production systems would use OAuth2/JWT |
| Data Integrity | SHA-256 (FIPS 180-4) | Detects any byte-level change to the dataset |
| Input Validation | Pydantic v2 field constraints (`ge`, `le`, `pattern`) | Rejects out-of-range or malformed values before they reach the ML models |
| Audit Logging | CSV append-only log with ISO-8601 timestamps | Supports traceability and post-incident review |

---

## 7. Performance Specification

| Metric | Target | Achieved |
|---|---|---|
| Random Forest Accuracy | > 85% | 96.7% |
| Random Forest F1 (weighted) | > 0.85 | 0.968 |
| Linear Regression MAE | < 5 vehicles | 1.03 vehicles |
| Linear Regression RMSE | < 8 vehicles | 1.31 vehicles |
| Incident Detector False Positive Rate | < 5% | 0.0% |
| API Response Time (per prediction) | < 200 ms | ~55 ms (measured via Flask test client) |
| Dataset Size | ≥ 10,000 records | 11,520 records |

---

## 8. Testing Strategy

| Test Type | Tool | Coverage |
|---|---|---|
| Unit tests | Custom assert-based suite (`tests/test_pipeline.py`), pytest-compatible | Simulator, preprocessing, optimizer, security, ML model training |
| Integration test | Flask test client (`app.test_client()`) | Full `/health`, `/predict`, `/signal`, `/metrics` request/response cycle, including auth rejection |
| Manual verification | `train.py` console output | End-to-end pipeline run with target-vs-actual comparison |

All 9 automated unit tests pass; the Flask integration test confirms
correct 401 rejection without a valid API key and correct 200 JSON
responses with one.

---

## 9. Deployment Notes

This is a local/demo-scale system designed for assessment purposes.
For a production ITS deployment, the following upgrades would be
required (noted here for completeness, not implemented in this
submission):

- Replace static API key with OAuth2/JWT and per-client rate limiting
- Replace CSV storage with a managed database (e.g. PostgreSQL/TimescaleDB)
- Replace simulated data with real sensor/RSU feeds via a message broker (e.g. MQTT/Kafka)
- Containerise (Docker) and deploy behind a load balancer for high availability
- Add model monitoring/drift detection for the ML components in production
