# Insurance Cost Prediction API

A REST API for predicting insurance costs based on personal information using machine learning.

This is an Industry 4.0 (IND4.0) project, demonstrating data-driven intelligence for healthcare insurance pricing and deployment-ready MLOps practices.

## Context

A Medical Insurance Company has released data for almost 1000 customers. Create a model that predicts the yearly medical cover cost. The data is voluntarily given by customers.

### Content

The dataset contains health-related parameters of the customers. Use them to build a model and also perform EDA on the same.

The Premium Price is in INR (₹) currency and showcases prices for a whole year.

### Inspiration

Help solve a crucial finance problem that would potentially impact many people and would help them make better decisions.

Don't forget to submit your EDAs and models in the task section. These will be keenly reviewed.

Note: This is a dummy dataset used for teaching and training purposes.

## Setup

1. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

2. Make sure you have the `Medicalpremium.csv` file in the same directory.

## Running the API

Start the API server:
```bash
python app.py
```

The API will be available at `http://localhost:5000`

## Deploying to Render

This repo includes `render.yaml` and a `Procfile` for one-click deployment.

### Quick Start (Dashboard)
- Create a new Web Service in Render from your Git repository.
- Render will detect Python, run `pip install -r requirements.txt`, and start with `python app.py`.
- Ensure the dataset file `Medicalpremium.csv` is committed to the repo (it is required at runtime).
 - The app pins Python runtime via `runtime.txt` to `python-3.10.13` for compatibility.
 - The app pins Python to 3.10.13 on Render via `PYTHON_VERSION=3.10.13` in `render.yaml` (and `.python-version`).

### Using render.yaml
- If you use Infrastructure as Code, Render will pick up `render.yaml` and configure:
  - Environment: Python
  - Build: `pip install --upgrade pip setuptools wheel && pip install -r requirements.txt`
  - Start: `python app.py`
  - Port is provided via `PORT` env var automatically; the app respects it.

After deploy, your service will be available at the Render-provided URL. CORS is enabled by default in this app.

## API Endpoints

### Health Check
- **GET** `/health`
  - Check if the API is running and if the model is loaded

### Make a Prediction
- **POST** `/predict`
  - Request body (JSON):
    ```json
    {
        "age": 48,
        "gender": 0,
        "bmi": 26.2,
        "smoker": 1,
        "alcohol_consumption": 1,
        "annual_income": 561000,
        "region": "urban",
        "pre_existing_conditions": 1
    }
    ```
  - Response (annual):
    ```json
    {
        "prediction": 133172.0,
        "formatted_prediction": "INR 133172.00 per year",
        "period": "annual",
        "status": "success",
        "input_data": {
            "age": 48,
            "gender": 0,
            "bmi": 26.2,
            "smoker": 1,
            "alcohol_consumption": 1,
            "annual_income": 561000,
            "region": "urban",
            "pre_existing_conditions": 1
        }
    }
    ```

### Retrain Model
- **POST** `/train`
  - Retrains the model with the current dataset
  - No request body needed
  - Response:
    ```json
    {
        "status": "success",
        "message": "Model retrained successfully"
    }
    ```

## Demonstration Video

📹 **Watch the API in action with Postman:**

<iframe src="https://drive.google.com/file/d/11-6erBpH7F0GoyrgCZcD4d64jbhtdXfr/preview" width="640" height="480" allow="autoplay"></iframe>

**Direct Link:** [View on Google Drive](https://drive.google.com/file/d/11-6erBpH7F0GoyrgCZcD4d64jbhtdXfr/view?usp=sharing)

This video demonstrates:
- How to test the API using Postman
- Making prediction requests with sample data
- Understanding the API response format
- Testing different health parameter combinations

## Example Usage with cURL

```bash
# Health check
curl http://localhost:5000/health

# Make a prediction
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"age":31,"sex":"female","bmi":25.74,"children":0,"smoker":"no","region":"southeast"}'
```

## Example Usage with Python

```python
import requests
import json

# Health check
response = requests.get("http://localhost:5000/health")
print(response.json())

# Make a prediction
data = {
  "age": 48,
  "gender": 0,
  "bmi": 26.2,
  "smoker": 1,
  "alcohol_consumption": 1,
  "annual_income": 561000,
  "region": "urban",
  "pre_existing_conditions": 1
}

response = requests.post(
    "http://localhost:5000/predict",
    headers={"Content-Type": "application/json"},
    data=json.dumps(data)
)

print(response.json())
```

## EDA: Generate Matplotlib Graphs

Run the EDA plotting script to save graphs to the `plots/` directory:

```bash
python eda_plots.py
```

It will generate:
- Histograms for numeric columns (e.g., `Age`, `Height`, `Weight`, `NumberOfMajorSurgeries`, `PremiumPrice`)
- Count plots for binary columns (e.g., `Diabetes`, `AnyChronicDiseases`, ...)
- Scatter plots of each feature vs `PremiumPrice`
- Boxplots of `PremiumPrice` grouped by each binary feature
- Correlation heatmap (`correlation_heatmap.png`)
- Pairwise scatter matrix for numeric columns
