# Ronin Blockchain User Classification

**ML Zoomcamp Midterm Project**  
**Author:** Jo$h  

*Predicting which Ronin blockchain users will remain active based on their historical transaction behavior using Machine Learning.*

---

## 📋 Table of Contents

- [Problem Description](#problem-description)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Features](#features)
- [Model Performance](#model-performance)
- [Installation](#installation)
- [Usage](#usage)
- [API Documentation](#api-documentation)
- [Docker Deployment](#docker-deployment)
- [Results](#results)
- [Future Improvements](#future-improvements)

---

## 🎯 Problem Description

### Business Problem
Identifying which blockchain users will remain active vs churn is critical for:
- **User retention strategies** - Focus resources on at-risk users
- **Community growth** - Understand what drives engagement
- **Resource allocation** - Prioritize high-value, consistent users

### Technical Problem
**Binary Classification:** Predict if a user will be active (5+ transactions) in the next 90 days based on their past 365 days of behavior.

**Target Variable:**
- `Good Trader` (1): User with ≥5 transactions in next 90 days
- `Bad Trader` (0): User with <5 transactions in next 90 days

---

## 📊 Dataset

### Data Source
- **Blockchain:** Ronin Network
- **Platform:** Dune Analytics
- **Time Period:** 
  - Training: 455 days ago → 90 days ago (365-day window)
  - Prediction: Last 90 days

### Dataset Statistics
- **Total Records:** 5,000 users
- **Class Distribution:** Perfectly balanced (2,500 Good, 2,500 Bad)
- **Historical Period:** 1,757 days of blockchain data available
- **Total Transactions:** 1.3+ billion transactions analyzed

### Features (5 numerical features)

| Feature | Description | Type |
|---------|-------------|------|
| `tx_count_365d` | Total transactions in past 365 days | Integer |
| `total_volume` | Total transaction volume in USD | Float |
| `active_weeks` | Number of weeks user was active | Integer |
| `avg_tx_value` | Average value per transaction | Float |
| `tx_per_active_week` | Transactions per active week | Float |

---

## 📁 Project Structure

```
ronin-users-classification/
├── data/
│   └── ronin_traders_dataset.csv          # Training dataset
├── notebooks/
│   └── 01_eda_and_training.ipynb          # EDA & model training
├── models/
│   ├── best_model_random_forest.pkl       # Trained model
│   ├── feature_names.pkl                  # Feature names
│   └── model_comparison_results.csv       # Performance metrics
├── visualizations/
│   ├── eda_analysis.png                   # EDA visualizations
│   ├── model_comparison.png               # Model performance
│   ├── confusion_matrix.png               # Confusion matrix
│   └── feature_importance.png             # Feature importance
├── src/
│   ├── predict.py                         # Prediction script
│   ├── app.py                             # Flask API
│   └── test_api.py                        # API tests
├── Dockerfile                              # Docker configuration
├── requirements.txt                        # Python dependencies
└── README.md                               # This file
```

---

## 🔍 Features

### Numerical Features

1. **tx_count_365d** - Transaction frequency indicator
2. **total_volume** - Economic activity measure  
3. **active_weeks** - Consistency/engagement metric
4. **avg_tx_value** - Transaction size indicator
5. **tx_per_active_week** - Activity intensity measure

### Feature Importance (from Random Forest)

| Feature | Importance | Interpretation |
|---------|-----------|----------------|
| total_volume | 30% | 💰 Most critical factor |
| active_weeks | 28% | 📅 Consistency matters |
| tx_count_365d | 22% | 🔢 Activity level important |
| avg_tx_value | 11% | 💵 Transaction size relevant |
| tx_per_active_week | 9% | ⚡ Frequency moderately important |

---

## 🏆 Model Performance

### Models Trained
1. **Random Forest** ⭐ (Best)
2. XGBoost
3. Decision Tree
4. Logistic Regression

### Best Model: Random Forest

| Metric | Score | Interpretation |
|--------|-------|----------------|
| **ROC-AUC** | **0.9646** | Outstanding discriminative ability |
| **Accuracy** | **91.4%** | Correct 914/1000 predictions |
| **Precision** | **90.4%** | 90.4% of "Good" predictions are correct |
| **Recall** | **92.6%** | Catches 92.6% of actual good traders |
| **F1-Score** | **91.5%** | Balanced precision/recall |

### Confusion Matrix (Test Set)

|                | Predicted Bad | Predicted Good |
|----------------|---------------|----------------|
| **Actual Bad** | 451 ✅ | 49 ❌ |
| **Actual Good** | 37 ❌ | 463 ✅ |

**Key Insight:** Only 86 misclassifications out of 1,000 test samples (8.6% error rate)

### Model Comparison

| Model | Accuracy | ROC-AUC |
|-------|----------|---------|
| Random Forest | 91.4% | 0.9646 |
| XGBoost | 90.4% | 0.9619 |
| Decision Tree | 87.8% | 0.9468 |
| Logistic Regression | 83.0% | 0.8889 |

---

## 🛠️ Installation

### Prerequisites
- Python 3.10+
- pip or conda
- Docker (optional, for containerization)

### Local Setup

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/ronin-trader-classification.git
cd ronin-trader-classification
```

2. **Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Verify installation**
```bash
python predict.py
```

---

## 💻 Usage

### 1. Standalone Prediction Script

```python
from predict import RoninTraderPredictor

# Initialize predictor
predictor = RoninTraderPredictor()

# Single prediction
trader = {
    'tx_count_365d': 150,
    'total_volume': 25.5,
    'active_weeks': 20,
    'avg_tx_value': 0.17,
    'tx_per_active_week': 7.5
}

result = predictor.predict_single(trader)
print(f"Prediction: {result['prediction']}")
print(f"Confidence: {result['confidence']:.2%}")
```

**Run the example:**
```bash
python predict.py
```

### 2. Flask API Service

**Start the API:**
```bash
python app.py
```

API will be available at `http://localhost:5000`

### 3. Run Tests

```bash
python test_api.py
```

---

## 📡 API Documentation

### Base URL
```
http://localhost:5000
```

### Endpoints

#### 1. Home - `GET /`
Get API information

**Response:**
```json
{
  "service": "Ronin Trader Classification API",
  "version": "1.0",
  "model": "Random Forest",
  "model_performance": {
    "accuracy": "91.4%",
    "roc_auc": "0.9646"
  }
}
```

#### 2. Health Check - `GET /health`
Check service status

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "features_loaded": true
}
```

#### 3. Get Features - `GET /features`
Get required feature information

**Response:**
```json
{
  "required_features": ["tx_count_365d", "total_volume", ...],
  "descriptions": {...},
  "example": {...}
}
```

#### 4. Single Prediction - `POST /predict`

**Request:**
```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "tx_count_365d": 150,
    "total_volume": 25.5,
    "active_weeks": 20,
    "avg_tx_value": 0.17,
    "tx_per_active_week": 7.5
  }'
```

**Response:**
```json
{
  "prediction": "Good Trader",
  "will_remain_active": true,
  "confidence": 0.92,
  "probability_good_trader": 0.92,
  "probability_bad_trader": 0.08,
  "input_features": {...}
}
```

#### 5. Batch Prediction - `POST /predict_batch`

**Request:**
```bash
curl -X POST http://localhost:5000/predict_batch \
  -H "Content-Type: application/json" \
  -d '{
    "traders": [
      {"tx_count_365d": 500, "total_volume": 100.0, ...},
      {"tx_count_365d": 10, "total_volume": 0.5, ...}
    ]
  }'
```

**Response:**
```json
{
  "predictions": [
    {
      "index": 0,
      "prediction": "Good Trader",
      "confidence": 0.95,
      ...
    }
  ],
  "summary": {
    "total": 2,
    "good_traders": 1,
    "bad_traders": 1,
    "percentage_good": 50.0
  }
}
```

---

## 🐳 Docker Deployment

### Build Docker Image

```bash
docker build -t ronin-trader-classifier .
```

### Run Container

```bash
docker run -p 5000:5000 ronin-trader-classifier
```

### Test Dockerized API

```bash
curl http://localhost:5000/health
```

### Docker Compose (Optional)

Create `docker-compose.yml`:
```yaml
version: '3.8'
services:
  api:
    build: .
    ports:
      - "5000:5000"
    environment:
      - PORT=5000
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:5000/health"]
      interval: 30s
      timeout: 3s
      retries: 3
```

Run with:
```bash
docker-compose up
```

---

## 📈 Results

### Key Findings

1. **Transaction volume is the strongest predictor** (30% importance)
   - Higher volume → Higher retention
   
2. **Consistency matters more than intensity** 
   - Active weeks (28%) > Transactions per week (9%)
   
3. **Model achieves 96.46% ROC-AUC**
   - Production-ready performance
   - Balanced precision and recall

### Business Impact

- **Churn Prediction:** Identify 92.6% of users at risk of leaving
- **Resource Optimization:** Focus retention efforts on predicted churners
- **Early Warning:** 90-day advance notice for intervention
- **Accuracy:** 91.4% correct predictions

---

## 🔮 Future Improvements

### Model Enhancements
- [ ] Add temporal features (time since first transaction, recent trend)
- [ ] Include token diversity (number of unique tokens traded)
- [ ] Add contract interaction features (DeFi usage)
- [ ] Implement online learning for continuous updates

### Technical Improvements
- [ ] Add model versioning
- [ ] Implement A/B testing framework
- [ ] Add monitoring and alerting
- [ ] Create CI/CD pipeline
- [ ] Add authentication to API

### Data Improvements
- [ ] Collect more granular data (hourly patterns)
- [ ] Include market conditions (RON price, gas fees)
- [ ] Add social features (unique counterparties)

---

## 📚 Technologies Used

- **Python 3.10** - Programming language
- **Pandas & NumPy** - Data manipulation
- **Scikit-learn** - Machine learning
- **XGBoost** - Gradient boosting
- **Flask** - Web framework
- **Docker** - Containerization
- **Dune Analytics** - Blockchain data
- **Jupyter** - Interactive development

---

## 📝 License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

---

## 🤝 Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Submit a pull request

---

## 📧 Contact

- X: [@defi__josh](https://x.com/defi__josh)
- LinkedIn: [Joshua Nwachukwu](https://www.linkedin.com/in/joshua-nwachukwu-1a0037188)
- Email: joshuatochinwachi@gmail.com

---

## 🙏 Acknowledgments

- **DataTalksClub** - ML Zoomcamp course
- **Dune Analytics** - Blockchain data platform
- **Ronin Network** - Blockchain infrastructure
- **Sky Mavis** - Ronin creators

---

**Built with ❤️ for the ML Zoomcamp Midterm Project**