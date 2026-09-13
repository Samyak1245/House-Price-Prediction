# California House Price Prediction

An end-to-end Machine Learning web application that predicts California house prices based on demographic, housing, and geographical features using the California Housing dataset. 

**[View Live Application](https://house-price-prediction-85ha.onrender.com/)**

## Tech Stack
* **Machine Learning & Data:** Python, Pandas, NumPy, Scikit-learn
* **Web Development:** Flask, HTML, CSS
* **Deployment:** Render, Git

## Quick Start

```bash
# Clone repository and navigate to project
git clone https://github.com/YOUR-USERNAME/YOUR-REPOSITORY.git
cd House-Price-Prediction

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the Flask server
python app.py
```
*Access the application locally at `http://127.0.0.1:5000/`*

## 🤖 Model Pipeline
The project handles end-to-end data processing, utilizing a regression model trained on 8 core features (e.g., Median Income, Population, Coordinates). User inputs are preprocessed through a saved `StandardScaler` (`scaling.pkl`) before generating predictions via the serialized model (`regmodel.pkl`).

## 🔮 Future Scope
* Hyperparameter tuning and algorithm comparisons to optimize accuracy.
* Docker containerization for standardized deployment.
* Enhanced UI/UX with interactive data visualizations.