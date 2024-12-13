# Stock Forecasting Using LSTM

This project demonstrates a stock price forecasting model built using LSTM (Long Short-Term Memory) networks. The project focuses on predicting Tesla's stock closing prices based on historical data.

## Features
- **Data Preprocessing**: Cleaned and normalized stock price data for better model performance.
- **LSTM Model**: Utilizes a Sequential LSTM-based architecture for time-series prediction.
- **Visualization**: Compares actual vs predicted stock prices for performance evaluation.

## Requirements
Ensure the following libraries are installed in your Python environment:

```bash
numpy
pandas
matplotlib
seaborn
scikit-learn
tensorflow
```

Install the libraries using pip:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn tensorflow
```

## Data Source
The dataset used in this project is Tesla stock data, which includes columns such as `Date`, `Open`, `High`, `Low`, `Close`, and `Volume`.

## Workflow

1. **Data Loading**
   The stock data is read from a CSV file and the `Close` column is used for prediction.

2. **Data Visualization**
   Initial plots of the stock price data provide insights into the trends and patterns.

3. **Data Preprocessing**
   - The data is normalized using `MinMaxScaler`.
   - Training and testing datasets are split (80% train, 20% test).
   - A sliding window approach is used to create sequences with 60 time steps for the LSTM model.

4. **Model Creation**
   The model includes:
   - Two LSTM layers
   - Dense layers for output
   - Optimized with `adam` optimizer and `mean squared error` loss function

5. **Model Training**
   The model is trained for 200 epochs to minimize the error.

6. **Testing and Prediction**
   - The model generates predictions on the test set.
   - Predictions are plotted against actual values to evaluate performance.

## How to Run
1. Clone this repository:
   ```bash
   git clone <repository_url>
   cd <repository_folder>
   ```
2. Ensure the dataset file (e.g., `TSLA.csv`) is in the project directory.
3. Run the Jupyter Notebook:
   ```bash
   jupyter notebook stock_forecasting.ipynb
   ```

## Example Output
The output includes a plot comparing actual stock prices with predicted prices for Tesla stock.

## Key Libraries Used
- **NumPy**: Numerical operations
- **Pandas**: Data manipulation
- **Matplotlib/Seaborn**: Data visualization
- **Scikit-learn**: Preprocessing and evaluation metrics
- **TensorFlow/Keras**: Building and training the LSTM model

## Results
- The model provides reasonable accuracy in predicting the trends of Tesla's closing stock prices.
- Example visualization shows the overlap between actual and predicted prices, demonstrating the model's capability.

## Future Improvements
- Experiment with more advanced architectures like GRU or Transformer models.
- Incorporate additional features like trading volume, moving averages, or external economic indicators.
- Fine-tune hyperparameters for better performance.

## License
This project is licensed under the MIT License. See the `LICENSE` file for details.

