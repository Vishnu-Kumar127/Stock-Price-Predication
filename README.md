# Stock Price Prediction

This repository contains a Flask-based web application for stock price prediction using pre-trained LSTM and GRU models. The app allows users to visualize recent stock prices, predict future prices based on sentiment scores, and evaluate model performance.

## Features

- **Stock Symbol Selection**: Choose from predefined stock options mapped to CSV files.
- **Data Visualization**: Display recent stock prices (past 30 days) dynamically downloaded using yfinance.
- **Stock Price Prediction**: Predict future stock prices using pre-trained LSTM or GRU models.
- **Model Selection**: Choose between LSTM and GRU models for predictions.
- **Performance Metrics**: Evaluate predictions with RMSE, MAE, and R² score.
- **Clean UI**: Left panel for inputs, right panel for results (charts, tables, metrics).

## Project Structure

```
Stock-Price-Prediction/
│── app.py               # Main Flask application
│── models/
│   ├── lstm_stock_model.h5  # Pre-trained LSTM model
│   ├── gru_stock_model.h5   # Pre-trained GRU model
│── data/
│   ├── stocks/             # CSV datasets
│   │   ├── GoogleWithSentiments.csv
│── static/
│   ├── style.css           # CSS for UI styling
│── templates/
│   ├── index.html          # Main UI template
│── requirements.txt        # Python dependencies
│── README.md               # Project documentation
```

## Installation

1. **Clone the repository**

   ```sh
   git clone https://github.com/Vishnu-Kumar127/Stock-Price-Predication.git
   cd Stock-Price-Predication
   ```

2. **Create a virtual environment and activate it**

   ```sh
   python -m venv venv
   venv/Scripts/activate  
   ```

3. **Install dependencies**

   ```sh
   pip install -r requirements.txt
   ```

4. **Run the Flask app**

   ```sh
   python app.py
   ```

5. **Access the app in the browser**

   ```
   http://127.0.0.1:5000
   ```

## Usage

1. **Select a stock** from the dropdown.
2. **Choose an action** (Recent Data, Predict, or Visualize).
3. If predicting, **select a model** (LSTM or GRU) and submit.
4. View results in the right panel (charts, predictions, or recent data table).

## Dependencies

- Flask
- TensorFlow/Keras
- Pandas
- NumPy
- Matplotlib
- yfinance

Install dependencies using:

```sh
pip install -r requirements.txt
```

## Future Enhancements

- Add more stock datasets.
- Improve UI with interactive visualizations.
- Deploy to a cloud server.

## License

This project is open-source and available under the MIT License.

## Author

[Vishnu Kumar](https://github.com/Vishnu-Kumar127)

