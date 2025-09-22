# Stock Price Reversal Analysis 

A comprehensive Streamlit web application for analyzing intraday stock price reversals, focusing on drawdown patterns and recovery strength for Indian stock market securities.

## Overview

This application helps traders and analysts understand intraday price movements by analyzing:
- **Maximum Drawdowns**: The percentage decline from opening price to intraday low
- **Reversal Strength**: The percentage recovery from intraday low to closing price
- **Recovery Patterns**: Statistical analysis of how often stocks recover from drawdowns

## Features

### Data Analysis
- **Multi-stock Analysis**: Analyze up to 9 pre-selected Indian stocks simultaneously
- **Flexible Time Ranges**: Choose from 1-5 years of historical data
- **High-Resolution Data**: 15-minute interval data for recent 60 days, daily data for longer periods
- **Real-time Data**: Fetches live data from Yahoo Finance

### Key Metrics
- **Drawdown Statistics**: Average, median, 75th and 90th percentile drawdowns
- **Reversal Analytics**: Recovery strength percentages and success rates
- **Recovery Rate**: Percentage of days with >50% recovery from drawdowns

### Supported Stocks
- **HDFC Bank** (HDFCBANK.NS)
- **Bajaj Finance** (BAJFINANCE.NS)
- **Adani Enterprises** (ADANIENT.NS)
- **Adani Ports** (ADANIPORTS.NS)
- **Shriram Finance** (SHRIRAMFIN.NS)
- **Tata Motors** (TATAMOTORS.NS)
- **IndusInd Bank** (INDUSINDBK.NS)
- **Bharat Electronics** (BEL.NS)
- **State Bank of India** (SBIN.NS)

### Visualizations
- **Distribution Plots**: Histogram and boxplot analysis of drawdown patterns
- **Time Series Analysis**: Monthly trends of drawdown and reversal strength
- **Comparative Analysis**: Side-by-side comparison of multiple stocks
- **Interactive Charts**: Plotly-powered interactive visualizations

### Interactive Features
- **Stock Selection**: Multi-select dropdown with friendly stock names
- **Time Period Slider**: Adjust analysis period from 1-5 years
- **High-Resolution Toggle**: Switch to 60-day high-resolution analysis
- **Raw Data Viewer**: Inspect underlying data for any selected stock

## Installation

### Prerequisites
- Python 3.7 or higher
- pip package manager

### Setup Instructions

1. **Clone the repository**
   ```bash
   git clone https://github.com/prachi-pandey-github/stock_price_reversal-.git
   cd stock_price_reversal-
   ```

2. **Install required packages**
   ```bash
   pip install streamlit yfinance pandas numpy matplotlib seaborn plotly
   ```

3. **Run the application**
   ```bash
   streamlit run streamlit_app.py
   ```

4. **Access the app**
   - Open your web browser and navigate to `http://localhost:8501`

## Dependencies

```
streamlit>=1.28.0
yfinance>=0.2.0
pandas>=1.5.0
numpy>=1.24.0
matplotlib>=3.6.0
seaborn>=0.12.0
plotly>=5.15.0
```

## Project Structure

```
stock_price_reversal-/
│
├── streamlit_app.py          # Main application file
├── README.md                 # Project documentation
└── requirements.txt          # Python dependencies (optional)
```

## Usage

### Basic Analysis
1. **Select Stocks**: Choose one or more stocks from the sidebar
2. **Set Time Period**: Use the slider to select analysis duration (1-5 years)
3. **View Results**: Analyze summary statistics, visualizations, and insights

### Advanced Features
- **High-Resolution Analysis**: Enable the checkbox for 15-minute interval data (last 60 days)
- **Time Series Exploration**: Select individual stocks for detailed monthly trend analysis
- **Raw Data Inspection**: Toggle raw data view to examine underlying calculations

### Key Metrics Explained
- **Drawdown %**: `(Open - Low) / Open × 100`
- **Reversal Strength %**: `(Close - Low) / (Open - Low) × 100`
- **Recovery Rate**: Percentage of trading days with >50% recovery

## Understanding the Analysis

### Drawdown Analysis
- **Purpose**: Identify how much stocks typically decline during the day
- **Interpretation**: Higher drawdowns indicate more intraday volatility
- **Usage**: Risk assessment and position sizing

### Reversal Strength Analysis
- **Purpose**: Measure recovery patterns from intraday lows
- **Interpretation**: Higher reversal strength suggests better recovery capability
- **Usage**: Entry/exit timing and trend reversal identification

### Recovery Rate
- **Purpose**: Determine consistency of bounce-back patterns
- **Interpretation**: Higher recovery rates indicate reliable reversal patterns
- **Usage**: Strategy validation and stock selection

## Features Breakdown

### User Interface
- **Responsive Design**: Works on desktop and mobile devices
- **Clean Layout**: Organized sections with clear visual hierarchy
- **Interactive Elements**: Real-time updates based on user selections
- **Professional Styling**: Custom CSS for enhanced visual appeal

### Data Processing
- **Robust Error Handling**: Graceful handling of data fetch failures
- **Data Validation**: Ensures data integrity and completeness
- **Caching**: Efficient data retrieval with 1-hour cache TTL
- **Multi-timeframe Support**: Automatic interval selection based on date range

### Performance Optimizations
- **Streamlit Caching**: Reduces redundant API calls
- **Progress Indicators**: Real-time feedback during data loading
- **Efficient Processing**: Optimized pandas operations for large datasets

## Technical Details

### Data Source
- **Provider**: Yahoo Finance via `yfinance` library
- **Coverage**: Indian stock market (NSE) securities
- **Frequency**: 15-minute intervals (recent) / Daily (historical)
- **Reliability**: Real-time data with automatic error handling

### Calculations
- All percentage calculations use opening price as the base
- Drawdowns are calculated as the maximum intraday decline
- Reversal strength measures the recovery from the lowest point
- Statistical measures include mean, median, and percentile analysis

## Limitations

- **Data Availability**: 15-minute data limited to last 60 days
- **Market Hours**: Analysis based on official trading hours only
- **Stock Universe**: Limited to 9 pre-selected Indian stocks
- **Internet Dependency**: Requires active internet connection for data fetch

