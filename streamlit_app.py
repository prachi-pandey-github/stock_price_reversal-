import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# Set page configuration
st.set_page_config(
    page_title="Stock Price Reversal Analysis",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {font-size: 2.5rem; color: #1f77b4; text-align: center; margin-bottom: 1rem;}
    .section-header {font-size: 1.8rem; color: #1f77b4; border-bottom: 2px solid #1f77b4; padding-bottom: 0.3rem; margin-top: 1.5rem;}
    .highlight {background-color: #f0f2f6; padding: 15px; border-radius: 5px; margin: 10px 0;}
    .metric-card {background-color: #f8f9fa; padding: 15px; border-radius: 5px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); margin: 5px;}
</style>
""", unsafe_allow_html=True)

# App title
st.markdown('<h1 class="main-header">Stock Price Reversal Analysis</h1>', unsafe_allow_html=True)
st.write("Analyze intraday price reversals: drawdowns and recovery patterns")

# Stock list
stock_list = [
    "HDFCBANK.NS", "BAJFINANCE.NS", "ADANIENT.NS", "ADANIPORTS.NS", "SHRIRAMFIN.NS",
    "TATAMOTORS.NS", "INDUSINDBK.NS", "BEL.NS", "SBIN.NS"
]
stock_names = {
    "HDFCBANK.NS": "HDFC Bank",
    "BAJFINANCE.NS": "Bajaj Finance",
    "ADANIENT.NS": "Adani Enterprises",
    "ADANIPORTS.NS": "Adani Ports",
    "SHRIRAMFIN.NS": "Shriram Finance",
    "TATAMOTORS.NS": "Tata Motors",
    "INDUSINDBK.NS": "IndusInd Bank",
    "BEL.NS": "Bharat Electronics",
    "SBIN.NS": "State Bank of India"
}

# Sidebar for user inputs
with st.sidebar:
    st.header("Configuration")
    selected_stocks = st.multiselect(
        "Select stocks to analyze:",
        options=stock_list,
        format_func=lambda x: stock_names[x],
        default=stock_list[:5]
    )
    
    years = st.slider("Select number of years of data:", 1, 5, 3)
    start_date = datetime.now() - timedelta(days=365*years)
    
    # Information about data availability
    days_requested = (datetime.now() - start_date).days
    if days_requested <= 60:
        st.info("📊 **15-minute interval data** will be used (available for last 60 days)")
    else:
        st.info("📈 **Daily interval data** will be used (15-minute data only available for last 60 days)")
    
    st.markdown("---")
    st.info("This app analyzes intraday price reversals by calculating:\n"
            "- Maximum Drawdown (% decline from open)\n"
            "- Reversal Strength (% recovery from low to close)")
    
    # Add option for recent high-resolution data
    if st.checkbox("Use recent 60-day high-resolution data instead"):
        years = 1
        start_date = datetime.now() - timedelta(days=60)
        st.success("Switched to 60-day period for 15-minute interval data")

# Main app functionality
@st.cache_data(ttl=3600)  # Cache data for 1 hour
def fetch_stock_data(stocks, start_date, end_date):
    """Fetch stock data with appropriate interval based on date range"""
    data_dict = {}
    failed_downloads = []
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Determine appropriate interval based on date range
    days_diff = (end_date - start_date).days
    if days_diff <= 60:
        interval = "15m"
        st.info(f"Using 15-minute interval data for {days_diff} days of data")
    else:
        interval = "1d"
        st.info(f"Using daily interval data for {days_diff} days of data (15m data only available for last 60 days)")
    
    for i, stock in enumerate(stocks):
        status_text.text(f"Fetching data for {stock_names[stock]}...")
        try:
            # Fetch data with determined interval
            data = yf.download(stock, start=start_date, end=end_date, interval=interval)
            if not data.empty:
                # Handle MultiIndex columns if present
                if isinstance(data.columns, pd.MultiIndex):
                    # Flatten MultiIndex columns
                    data.columns = data.columns.get_level_values(0)
                data_dict[stock] = data
                st.success(f"✓ Successfully loaded {stock_names[stock]}")
            else:
                st.warning(f"No data found for {stock_names[stock]}")
                failed_downloads.append(stock)
        except Exception as e:
            st.error(f"Failed to fetch data for {stock_names[stock]}: {str(e)}")
            failed_downloads.append(stock)
        progress_bar.progress((i + 1) / len(stocks))
    
    if failed_downloads:
        st.warning(f"Failed to download data for: {', '.join([stock_names[s] for s in failed_downloads])}")
    
    status_text.empty()
    progress_bar.empty()
    return data_dict

@st.cache_data
def preprocess_data(data_dict):
    """Process data to calculate daily Open, Low, Close metrics"""
    daily_data = {}
    
    for stock, df in data_dict.items():
        if df.empty:
            continue
        
        try:
            # Ensure we have a clean DataFrame copy
            df_clean = df.copy()
            
            # Handle MultiIndex columns if still present
            if isinstance(df_clean.columns, pd.MultiIndex):
                df_clean.columns = df_clean.columns.get_level_values(0)
            
            # Standardize column names (remove any extra spaces, ensure proper case)
            df_clean.columns = df_clean.columns.str.strip()
            column_mapping = {}
            for col in df_clean.columns:
                col_lower = col.lower()
                if 'open' in col_lower:
                    column_mapping[col] = 'Open'
                elif 'high' in col_lower:
                    column_mapping[col] = 'High'
                elif 'low' in col_lower:
                    column_mapping[col] = 'Low'
                elif 'close' in col_lower:
                    column_mapping[col] = 'Close'
                elif 'volume' in col_lower:
                    column_mapping[col] = 'Volume'
            
            df_clean = df_clean.rename(columns=column_mapping)
            
            # Check if data is already daily or needs resampling
            time_diff = df_clean.index[1] - df_clean.index[0] if len(df_clean) > 1 else None
            if time_diff and time_diff < pd.Timedelta(hours=23):
                # Intraday data - resample to daily
                daily_df = pd.DataFrame()
                daily_df['Open'] = df_clean['Open'].resample('D').first()
                daily_df['Low'] = df_clean['Low'].resample('D').min()
                daily_df['Close'] = df_clean['Close'].resample('D').last()
                daily_df['High'] = df_clean['High'].resample('D').max()
            else:
                # Already daily data
                daily_df = df_clean.copy()
            
            # Ensure we have the required columns
            required_columns = ['Open', 'Low', 'Close', 'High']
            if not all(col in daily_df.columns for col in required_columns):
                st.warning(f"Missing required columns for {stock_names[stock]}. Available columns: {list(daily_df.columns)}")
                continue
            
            # Remove days with missing data
            daily_df = daily_df[required_columns].dropna()
            
            if daily_df.empty:
                st.warning(f"No valid data after preprocessing for {stock_names[stock]}")
                continue
            
            # Calculate metrics - ensure we're working with Series, not DataFrames
            daily_df = daily_df.copy()  # Ensure we have a proper DataFrame
            daily_df['Drawdown_$'] = daily_df['Open'] - daily_df['Low']
            daily_df['Drawdown_%'] = (daily_df['Drawdown_$'] / daily_df['Open']) * 100
            daily_df['Recovery_$'] = daily_df['Close'] - daily_df['Low']
            
            # Handle cases where drawdown is 0 (avoid division by zero)
            daily_df['Reversal_Strength_%'] = np.where(
                daily_df['Drawdown_$'] == 0, 
                100, 
                (daily_df['Recovery_$'] / daily_df['Drawdown_$']) * 100
            )
            
            daily_data[stock] = daily_df
            
        except Exception as e:
            st.error(f"Error processing data for {stock_names[stock]}: {str(e)}")
            continue
    
    return daily_data

@st.cache_data
def calculate_statistics(daily_data):
    """Calculate summary statistics for each stock"""
    stats_data = []
    
    for stock, df in daily_data.items():
        if df.empty:
            continue
            
        stats = {
            'Stock': stock_names[stock],
            'Ticker': stock,
            'Days': len(df),
            'Avg_Drawdown_%': df['Drawdown_%'].mean(),
            'Median_Drawdown_%': df['Drawdown_%'].median(),
            'P75_Drawdown_%': np.percentile(df['Drawdown_%'], 75),
            'P90_Drawdown_%': np.percentile(df['Drawdown_%'], 90),
            'Avg_Reversal_%': df['Reversal_Strength_%'].mean(),
            'Median_Reversal_%': df['Reversal_Strength_%'].median(),
            'P75_Reversal_%': np.percentile(df['Reversal_Strength_%'], 75),
            'P90_Reversal_%': np.percentile(df['Reversal_Strength_%'], 90),
            'Recovery_Rate': (df['Reversal_Strength_%'] > 50).mean() * 100  # % of days with >50% recovery
        }
        stats_data.append(stats)
    
    return pd.DataFrame(stats_data)

# Main app execution
if selected_stocks:
    end_date = datetime.now()
    
    # Fetch and process data
    with st.spinner("Downloading and processing stock data..."):
        stock_data = fetch_stock_data(selected_stocks, start_date, end_date)
        daily_data = preprocess_data(stock_data)
        stats_df = calculate_statistics(daily_data)
    
    # Display summary statistics
    st.markdown('<h2 class="section-header">Summary Statistics</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Drawdown Statistics**")
        drawdown_cols = ['Stock', 'Avg_Drawdown_%', 'Median_Drawdown_%', 'P75_Drawdown_%', 'P90_Drawdown_%']
        drawdown_df = stats_df[drawdown_cols].copy()
        drawdown_df.columns = ['Stock', 'Average', 'Median', '75th Percentile', '90th Percentile']
        st.dataframe(drawdown_df.style.format({
            'Average': '{:.2f}%',
            'Median': '{:.2f}%',
            '75th Percentile': '{:.2f}%',
            '90th Percentile': '{:.2f}%'
        }), use_container_width=True)
    
    with col2:
        st.markdown("**Reversal Statistics**")
        reversal_cols = ['Stock', 'Avg_Reversal_%', 'Median_Reversal_%', 'P75_Reversal_%', 'P90_Reversal_%', 'Recovery_Rate']
        reversal_df = stats_df[reversal_cols].copy()
        reversal_df.columns = ['Stock', 'Average', 'Median', '75th Percentile', '90th Percentile', 'Recovery Rate']
        st.dataframe(reversal_df.style.format({
            'Average': '{:.2f}%',
            'Median': '{:.2f}%',
            '75th Percentile': '{:.2f}%',
            '90th Percentile': '{:.2f}%',
            'Recovery Rate': '{:.2f}%'
        }), use_container_width=True)
    
    # Visualizations
    st.markdown('<h2 class="section-header">Visualizations</h2>', unsafe_allow_html=True)
    
    # Drawdown distribution
    st.markdown("**Distribution of Maximum Drawdowns**")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    all_drawdowns = []
    for stock in selected_stocks:
        if stock in daily_data:
            drawdowns = daily_data[stock]['Drawdown_%'].dropna()
            all_drawdowns.extend(zip([stock_names[stock]] * len(drawdowns), drawdowns))
    
    if all_drawdowns:
        drawdown_df = pd.DataFrame(all_drawdowns, columns=['Stock', 'Drawdown'])
        
        # Histogram with KDE
        sns.histplot(data=drawdown_df, x='Drawdown', hue='Stock', element='step', stat='density', common_norm=False, ax=axes[0])
        axes[0].set_title('Distribution of Drawdowns by Stock')
        axes[0].set_xlabel('Drawdown (%)')
        
        # Boxplot
        sns.boxplot(data=drawdown_df, x='Stock', y='Drawdown', ax=axes[1])
        axes[1].set_title('Drawdown Distribution by Stock')
        axes[1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        st.pyplot(fig)
    else:
        st.warning("No data available for visualization")
    
    # Reversal strength over time
    st.markdown("**Reversal Strength Over Time**")
    
    # Select a stock for time series analysis
    selected_stock = st.selectbox(
        "Select stock for time series analysis:",
        options=selected_stocks,
        format_func=lambda x: stock_names[x]
    )
    
    if selected_stock in daily_data:
        df = daily_data[selected_stock].copy()
        df['Date'] = df.index
        df['Month'] = df.index.to_period('M').astype(str)
        
        # Calculate monthly averages
        monthly_avg = df.groupby('Month').agg({
            'Drawdown_%': 'mean',
            'Reversal_Strength_%': 'mean'
        }).reset_index()
        
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        
        fig.add_trace(
            go.Scatter(x=monthly_avg['Month'], y=monthly_avg['Drawdown_%'], name="Drawdown (%)", line=dict(color='red')),
            secondary_y=False,
        )
        
        fig.add_trace(
            go.Scatter(x=monthly_avg['Month'], y=monthly_avg['Reversal_Strength_%'], name="Reversal Strength (%)", line=dict(color='green')),
            secondary_y=True,
        )
        
        fig.update_layout(
            title=f"Monthly Average Drawdown and Reversal Strength for {stock_names[selected_stock]}",
            xaxis_title="Month",
            hovermode="x unified"
        )
        
        fig.update_yaxes(title_text="Drawdown (%)", secondary_y=False)
        fig.update_yaxes(title_text="Reversal Strength (%)", secondary_y=True)
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Insights section
    st.markdown('<h2 class="section-header">Key Insights</h2>', unsafe_allow_html=True)
    
    if not stats_df.empty:
        # Find stock with highest average drawdown
        max_drawdown_stock = stats_df.loc[stats_df['Avg_Drawdown_%'].idxmax()]
        
        # Find stock with highest reversal strength
        max_reversal_stock = stats_df.loc[stats_df['Avg_Reversal_%'].idxmax()]
        
        # Find stock with highest recovery rate
        max_recovery_stock = stats_df.loc[stats_df['Recovery_Rate'].idxmax()]
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric(
                label="Highest Average Drawdown",
                value=f"{max_drawdown_stock['Stock']}",
                delta=f"{max_drawdown_stock['Avg_Drawdown_%']:.2f}%"
            )
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric(
                label="Highest Reversal Strength",
                value=f"{max_reversal_stock['Stock']}",
                delta=f"{max_reversal_stock['Avg_Reversal_%']:.2f}%"
            )
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col3:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric(
                label="Highest Recovery Rate",
                value=f"{max_recovery_stock['Stock']}",
                delta=f"{max_recovery_stock['Recovery_Rate']:.2f}%"
            )
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Generate insights
        st.markdown('<div class="highlight">', unsafe_allow_html=True)
        st.write("**Insights:**")
        st.write(f"- {max_drawdown_stock['Stock']} has the highest average intraday drawdown at {max_drawdown_stock['Avg_Drawdown_%']:.2f}%")
        st.write(f"- {max_reversal_stock['Stock']} shows the strongest reversal pattern, recovering {max_reversal_stock['Avg_Reversal_%']:.2f}% of drawdowns on average")
        st.write(f"- {max_recovery_stock['Stock']} has the highest recovery rate, with {max_recovery_stock['Recovery_Rate']:.2f}% of days showing more than 50% recovery")
        
        # Compare Adani stocks if available
        adani_stocks = [s for s in selected_stocks if 'ADANI' in s]
        if len(adani_stocks) >= 2:
            st.write("- Adani group stocks show high volatility with significant drawdowns but also strong reversal tendencies")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Raw data option
    if st.checkbox("Show raw data"):
        selected_raw_stock = st.selectbox(
            "Select stock to view raw data:",
            options=selected_stocks,
            format_func=lambda x: stock_names[x]
        )
        if selected_raw_stock in daily_data:
            st.dataframe(daily_data[selected_raw_stock].tail(20))
    
else:
    st.info("Please select at least one stock from the sidebar to begin analysis.")

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center'>
        <p>Stock Price Reversal Analysis App | Data sourced from Yahoo Finance</p>
    </div>
    """,
    unsafe_allow_html=True
)