import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class StockRecoveryAnalyzer:
    def __init__(self):
        self.data_source = 'yfinance'  # Only real market data from yfinance
        
    def fetch_historical_data(self, symbol, period="7d", interval="1m"):
        """
        Fetch comprehensive historical 1-minute interval data including:
        • Date-Time
        • Open Price  
        • High Price
        • Low Price
        • Close Price
        • Volume
        """
        try:
            if not symbol.endswith('.NS'):
                symbol += '.NS'
            
            print(f"Fetching {period} of {interval} data for {symbol}...")
            print(f"Required data fields: Date-Time, Open, High, Low, Close, Volume")
            print(f"📅 Extended Analysis Period: {period} - Capturing more market volatility...")
            
            ticker = yf.Ticker(symbol)
            data = ticker.history(period=period, interval=interval)
            
            if data.empty:
                print(f"No data returned for {symbol}")
                return None
                
            data.reset_index(inplace=True)
            
            # Ensure proper datetime column
            if 'Datetime' in data.columns:
                data['Datetime'] = pd.to_datetime(data['Datetime'])
            elif 'Date' in data.columns:
                data['Datetime'] = pd.to_datetime(data['Date'])
                data.drop('Date', axis=1, inplace=True)
            else:
                print("No datetime column found")
                return None
            
            # Verify all required OHLCV data fields are present
            required_fields = ['Open', 'High', 'Low', 'Close', 'Volume']
            missing_fields = [field for field in required_fields if field not in data.columns]
            
            if missing_fields:
                print(f"Warning: Missing required fields: {missing_fields}")
                return None
            
            print(f"Successfully fetched {len(data)} records for {symbol}")
            print(f"Data fields confirmed: {list(data.columns)}")
            print(f"Date range: {data['Datetime'].min()} to {data['Datetime'].max()}")
            
            return data
            
        except Exception as e:
            print(f"Error fetching data for {symbol}: {e}")
            return None

    
    def calculate_peak_price(self, data, lookback_period, price_type, current_idx):
        """
        Calculate peak price within lookback period
        """
        start_idx = max(0, current_idx - lookback_period)
        if start_idx >= current_idx:
            return data.iloc[current_idx][['Open', 'High', 'Low', 'Close']].max()
            
        window_data = data.iloc[start_idx:current_idx]
        
        if price_type == "Close":
            return window_data['Close'].max()
        elif price_type == "High/Low":
            return window_data['High'].max()
        else:
            raise ValueError("Price_Type_For_Peak_Trough_Drop must be 'Close' or 'High/Low'")
    
    def detect_drop_events(self, data, lookback_period, drop_threshold, recovery_target, 
                          price_type, min_drop_duration):
        """
        Algorithm Implementation as per Requirements:
        1. Iterate through historical data minute by minute
        2. Identify Peak_Price within Lookback_Period_For_Peak  
        3. Detect Drop Event when price falls below Peak_Price * (1 - Drop_Threshold_Percentage / 100)
        4. Ensure drop sustains for at least Minimum_Drop_Event_Duration
        5. Identify Trough_Price as lowest point after drop
        """
        print(f"  Analyzing drop threshold: {drop_threshold}%")
        
        drop_events = []
        current_event = None
        
        # Step 1: Iterate through historical data minute by minute
        for i in range(lookback_period, len(data)):
            
            # Step 2: Identify Peak_Price within Lookback_Period_For_Peak
            peak_price = self.calculate_peak_price(data, lookback_period, price_type, i)
            
            # Current price for comparison
            if price_type == "Close":
                current_price = data.iloc[i]['Close']
            else:
                current_price = data.iloc[i]['Low']
            
            # Step 3: Detect Drop Event threshold
            drop_threshold_price = peak_price * (1 - drop_threshold / 100)
            
            # Debug output for first few iterations
            if i < lookback_period + 5:
                timestamp = data.iloc[i]['Datetime']
                print(f"    Minute {i} ({timestamp}): Peak={peak_price:.2f}, Current={current_price:.2f}, Threshold={drop_threshold_price:.2f}")
            
            # Step 3: Check if Drop Event occurs
            if current_price <= drop_threshold_price:
                if current_event is None:
                    # New drop event detected
                    current_event = {
                        'peak_price': peak_price,
                        'drop_start_idx': i,
                        'trough_price': current_price,
                        'trough_idx': i,
                        'duration': 1
                    }
                else:
                    # Continue existing drop event
                    current_event['duration'] += 1
                    # Step 5: Update Trough_Price (lowest point after drop)
                    if current_price < current_event['trough_price']:
                        current_event['trough_price'] = current_price
                        current_event['trough_idx'] = i
            else:
                # Price recovered above threshold
                if current_event is not None:
                    # Step 4: Ensure drop sustains for at least Minimum_Drop_Event_Duration
                    if current_event['duration'] >= min_drop_duration:
                        # Valid drop event - calculate recovery target
                        total_price_drop = current_event['peak_price'] - current_event['trough_price']
                        current_event['total_price_drop'] = total_price_drop
                        # Step 6: Calculate Target_Recovery_Price
                        current_event['recovery_target_price'] = current_event['trough_price'] + (total_price_drop * (recovery_target / 100))
                        drop_events.append(current_event.copy())
                    
                    current_event = None
            
            # Progress indicator
            if i % 1000 == 0:
                print(f"    Processed minute {i}/{len(data)}, found {len(drop_events)} valid events...")
        
        # Handle any ongoing event at end of data
        if current_event is not None and current_event['duration'] >= min_drop_duration:
            total_price_drop = current_event['peak_price'] - current_event['trough_price']
            current_event['total_price_drop'] = total_price_drop
            current_event['recovery_target_price'] = current_event['trough_price'] + (total_price_drop * (recovery_target / 100))
            drop_events.append(current_event.copy())
        
        print(f"    Final analysis: {len(drop_events)} valid drop events detected")
        return drop_events
    
    def check_recovery(self, data, drop_event, price_type, max_lookahead=2880):
        """
        Algorithm Steps 7-8:
        7. Monitor subsequent intervals until price touches or exceeds Target_Recovery_Price
        8. Record event as Successful Recovery Event or failed event
        Max lookahead of 2880 minutes (2 days) for 1-minute data
        """
        trough_idx = drop_event['trough_idx']
        target_recovery_price = drop_event['recovery_target_price']
        
        # Step 7: Monitor subsequent intervals
        lookahead_limit = min(trough_idx + max_lookahead, len(data))
        
        for j in range(trough_idx + 1, lookahead_limit):
            if price_type == "Close":
                check_price = data.iloc[j]['Close']
            else:
                check_price = data.iloc[j]['High']
            
            # Step 7: Check if price touches or exceeds Target_Recovery_Price
            if check_price >= target_recovery_price:
                recovery_time_minutes = j - trough_idx
                # Step 8: Record as Successful Recovery Event
                return True, recovery_time_minutes
        
        # Step 8: Record as failed event (no recovery within lookahead period)
        return False, None
    
    def analyze_stock_recovery(self, symbol, lookback_period, drop_thresholds, 
                             recovery_target, price_type, min_drop_duration):
        """
        Complete analysis for a single stock
        """
        print(f"\n{'='*60}")
        print(f"ANALYZING: {symbol}")
        print(f"{'='*60}")
        
        # Use real historical data from yfinance - Extended period for better volatility capture
        # Note: Yahoo Finance limits 1m data to ~8 days, so use 5m intervals for 60-day analysis
        data = self.fetch_historical_data(symbol, period="60d", interval="5m")
        
        if data is None or len(data) == 0:
            print(f"No data available for {symbol}")
            return []
        
        # Detailed data validation for 5-minute OHLCV data
        print("\n📊 5-MINUTE DATA VALIDATION:")
        print(f"Total records: {len(data)}")
        print(f"Date range: {data['Datetime'].min()} to {data['Datetime'].max()}")
        print(f"OHLCV Data Summary:")
        print(f"  • Open: {data['Open'].min():.2f} - {data['Open'].max():.2f}")
        print(f"  • High: {data['High'].min():.2f} - {data['High'].max():.2f}")  
        print(f"  • Low: {data['Low'].min():.2f} - {data['Low'].max():.2f}")
        print(f"  • Close: {data['Close'].min():.2f} - {data['Close'].max():.2f}")
        print(f"  • Volume: {data['Volume'].min():,} - {data['Volume'].max():,}")
        
        # Sample data preview
        print("\n📋 Sample 5-minute records:")
        sample_cols = ['Datetime', 'Open', 'High', 'Low', 'Close', 'Volume']
        print(data[sample_cols].head(3).to_string(index=False))
        
        results = []
        
        for drop_threshold in drop_thresholds:
            drop_events = self.detect_drop_events(
                data, lookback_period, drop_threshold, recovery_target,
                price_type, min_drop_duration
            )
            
            total_drop_events = len(drop_events)
            successful_recoveries = 0
            recovery_times = []
            
            print(f"    Checking recovery for {total_drop_events} events...")
            
            for event_idx, event in enumerate(drop_events):
                print(f"      Event {event_idx+1}: Drop from {event['peak_price']:.2f} to {event['trough_price']:.2f} "
                      f"({(event['peak_price']-event['trough_price'])/event['peak_price']*100:.1f}%)")
                print(f"      Recovery target: {event['recovery_target_price']:.2f}")
                
                recovered, recovery_time = self.check_recovery(data, event, price_type)
                if recovered:
                    successful_recoveries += 1
                    recovery_times.append(recovery_time)
                    print(f"      ✓ Recovery achieved in {recovery_time} minutes")
                else:
                    print(f"      ✗ No recovery within lookahead period")
            
            recovery_probability = (successful_recoveries / total_drop_events * 100) if total_drop_events > 0 else 0
            avg_recovery_time = np.mean(recovery_times) if recovery_times else None
            
            results.append({
                'Stock Symbol': symbol,
                'Drop Threshold (%)': drop_threshold,
                'Total Drop Events Observed': total_drop_events,
                'Successful Recovery Events': successful_recoveries,
                'Recovery Probability (%)': round(recovery_probability, 2)
            })
        
        return results

def validate_configuration(config):
    """
    Validate user-configurable parameters to ensure correct types and ranges
    """
    print("🔍 VALIDATING CONFIGURATION PARAMETERS:")
    
    # 1. Validate Lookback_Period_For_Peak
    if not isinstance(config['LOOKBACK_PERIOD_FOR_PEAK'], int) or config['LOOKBACK_PERIOD_FOR_PEAK'] <= 0:
        raise ValueError("Lookback_Period_For_Peak must be a positive integer")
    print(f"✅ Lookback_Period_For_Peak: {config['LOOKBACK_PERIOD_FOR_PEAK']} intervals")
    
    # 2. Validate Drop_Thresholds_Percentage
    if not isinstance(config['DROP_THRESHOLDS_PERCENTAGE'], list):
        raise ValueError("Drop_Thresholds_Percentage must be a list of doubles")
    for threshold in config['DROP_THRESHOLDS_PERCENTAGE']:
        if not isinstance(threshold, (int, float)) or threshold <= 0 or threshold >= 100:
            raise ValueError("Each drop threshold must be between 0 and 100 percent")
    print(f"✅ Drop_Thresholds_Percentage: {config['DROP_THRESHOLDS_PERCENTAGE']}%")
    
    # 3. Validate Recovery_Target_Percentage
    if not isinstance(config['RECOVERY_TARGET_PERCENTAGE'], (int, float)) or config['RECOVERY_TARGET_PERCENTAGE'] <= 0:
        raise ValueError("Recovery_Target_Percentage must be a positive number")
    print(f"✅ Recovery_Target_Percentage: {config['RECOVERY_TARGET_PERCENTAGE']}%")
    
    # 4. Validate Price_Type_For_Peak_Trough_Drop
    valid_price_types = ["Close", "High/Low"]
    if config['PRICE_TYPE_FOR_PEAK_TROUGH_DROP'] not in valid_price_types:
        raise ValueError(f"Price_Type_For_Peak_Trough_Drop must be one of: {valid_price_types}")
    print(f"✅ Price_Type_For_Peak_Trough_Drop: {config['PRICE_TYPE_FOR_PEAK_TROUGH_DROP']}")
    
    # 5. Validate Minimum_Drop_Event_Duration
    if not isinstance(config['MINIMUM_DROP_EVENT_DURATION'], int) or config['MINIMUM_DROP_EVENT_DURATION'] <= 0:
        raise ValueError("Minimum_Drop_Event_Duration must be a positive integer")
    print(f"✅ Minimum_Drop_Event_Duration: {config['MINIMUM_DROP_EVENT_DURATION']} intervals")
    
    print("✅ All configuration parameters validated successfully!\n")
    return True

def main():
    """
    Main execution function with validated configuration parameters
    """
    # ================================================================
    # KEY USER-CONFIGURABLE PARAMETERS
    # ================================================================
    CONFIG = {
        # 1. Lookback_Period_For_Peak (Integer, e.g., 60 days)
        # Defines the window to search for the highest price preceding a potential drop
        # For 1-minute data: 60 days = 60 * 375 minutes (trading minutes per day) = 22,500 minutes
        # Current setting: 1440 minutes = 24 hours
        'LOOKBACK_PERIOD_FOR_PEAK': 1440,
        
        # 2. Drop_Thresholds_Percentage (List of Doubles, e.g., [10.0, 15.0, 20.0])
        # The specific percentage drops from Peak Price that trigger a "drop event" for analysis
        'DROP_THRESHOLDS_PERCENTAGE': [10.0, 15.0, 20.0],
        
        # 3. Recovery_Target_Percentage (Double, e.g., 40.0)
        # The percentage recovery desired from Trough Price relative to Total Drop amount
        'RECOVERY_TARGET_PERCENTAGE': 40.0,
        
        # 4. Price_Type_For_Peak_Trough_Drop (String, e.g., "Close", "High/Low")
        # "Close": Uses only Close Price for peak/trough/drop detection
        # "High/Low": Considers High and Low prices for intraday extremes
        'PRICE_TYPE_FOR_PEAK_TROUGH_DROP': "High/Low",
        
        # 5. Minimum_Drop_Event_Duration (Integer, e.g., 1 interval)
        # Minimum number of intervals price must be below Peak Price for valid drop event
        # Avoids noise from single-interval fluctuations
        'MINIMUM_DROP_EVENT_DURATION': 5,
        
        # Additional Configuration
        'STOCKS_TO_ANALYZE': ['ADANIENT', 'SHRIRAMFIN']  # Target stocks for analysis
    }
    
    # Display and validate configuration before proceeding
    display_configuration(CONFIG)
    validate_configuration(CONFIG)
    
    analyzer = StockRecoveryAnalyzer()
    all_results = []
    
    for stock in CONFIG['STOCKS_TO_ANALYZE']:
        try:
            results = analyzer.analyze_stock_recovery(
                symbol=stock,
                lookback_period=CONFIG['LOOKBACK_PERIOD_FOR_PEAK'],
                drop_thresholds=CONFIG['DROP_THRESHOLDS_PERCENTAGE'],
                recovery_target=CONFIG['RECOVERY_TARGET_PERCENTAGE'],
                price_type=CONFIG['PRICE_TYPE_FOR_PEAK_TROUGH_DROP'],
                min_drop_duration=CONFIG['MINIMUM_DROP_EVENT_DURATION']
            )
            all_results.extend(results)
        except Exception as e:
            print(f"Error analyzing {stock}: {e}")
            continue
    
    if all_results:
        results_df = pd.DataFrame(all_results)
        
        print("\n" + "="*100)
        print("STOCK RECOVERY PROBABILITY ANALYSIS RESULTS")
        print("="*100)
        print(results_df.to_string(index=False))
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"stock_recovery_analysis_{timestamp}.csv"
        results_df.to_csv(filename, index=False)
        print(f"\nResults saved to: {filename}")
        
        # Enhanced summary
        print("\nDETAILED SUMMARY:")
        print("="*60)
        for threshold in CONFIG['DROP_THRESHOLDS_PERCENTAGE']:
            threshold_data = results_df[results_df['Drop Threshold (%)'] == threshold]
            total_events = threshold_data['Total Drop Events Observed'].sum()
            successful_events = threshold_data['Successful Recovery Events'].sum()
            avg_prob = threshold_data['Recovery Probability (%)'].mean()
            
            print(f"{threshold}% drops: {successful_events}/{total_events} recovered (avg {avg_prob:.1f}%)")
        
        return results_df
    else:
        print("No results generated!")
        return None

def display_configuration(config):
    """
    Display current configuration parameters in a clear format
    """
    print("⚙️  CURRENT CONFIGURATION:")
    print("-" * 60)
    print(f"1. Lookback_Period_For_Peak: {config['LOOKBACK_PERIOD_FOR_PEAK']} intervals")
    print(f"   └─ Window to search for highest price preceding drop")
    print(f"2. Drop_Thresholds_Percentage: {config['DROP_THRESHOLDS_PERCENTAGE']}%")
    print(f"   └─ Percentage drops that trigger drop event analysis")
    print(f"3. Recovery_Target_Percentage: {config['RECOVERY_TARGET_PERCENTAGE']}%")
    print(f"   └─ Desired recovery from trough relative to total drop")
    print(f"4. Price_Type_For_Peak_Trough_Drop: {config['PRICE_TYPE_FOR_PEAK_TROUGH_DROP']}")
    print(f"   └─ Price data used for peak/trough/drop detection")
    print(f"5. Minimum_Drop_Event_Duration: {config['MINIMUM_DROP_EVENT_DURATION']} intervals")
    print(f"   └─ Minimum duration below peak for valid drop event")
    print(f"6. Stocks_To_Analyze: {config['STOCKS_TO_ANALYZE']}")
    print(f"   └─ Target stocks for recovery analysis")
    print("-" * 60)

def create_custom_config(lookback_period=1440, drop_thresholds=[10.0, 15.0, 20.0], 
                        recovery_target=40.0, price_type="High/Low", min_duration=5, 
                        stocks=['ADANIENT', 'SHRIRAMFIN']):
    """
    Helper function to create custom configuration with different parameters
    
    Args:
        lookback_period (int): Window to search for peak (e.g., 60 days = 22500 minutes)
        drop_thresholds (list): Percentage drops to analyze (e.g., [10.0, 15.0, 20.0])
        recovery_target (float): Target recovery percentage (e.g., 40.0)
        price_type (str): "Close" or "High/Low" for price detection
        min_duration (int): Minimum intervals for valid drop event
        stocks (list): List of stock symbols to analyze
    
    Returns:
        dict: Custom configuration dictionary
    """
    return {
        'LOOKBACK_PERIOD_FOR_PEAK': lookback_period,
        'DROP_THRESHOLDS_PERCENTAGE': drop_thresholds,
        'RECOVERY_TARGET_PERCENTAGE': recovery_target,
        'PRICE_TYPE_FOR_PEAK_TROUGH_DROP': price_type,
        'MINIMUM_DROP_EVENT_DURATION': min_duration,
        'STOCKS_TO_ANALYZE': stocks,
        'USE_SAMPLE_DATA': False
    }

if __name__ == "__main__":
    print("STOCK RECOVERY ANALYZER - EXTENDED PERIOD IMPLEMENTATION")
    print("="*80)
    print("📊 INPUT DATA REQUIREMENTS:")
    print("Historical 5-Minute Price Data including:")
    print("• Date-Time")
    print("• Open Price") 
    print("• High Price")
    print("• Low Price")
    print("• Close Price")
    print("• Volume")
    print("\n📅 EXTENDED ANALYSIS PERIOD: 60 days (5-min intervals) for better volatility capture")
    print("   Note: Yahoo Finance limits 1-minute data to ~8 days maximum")
    print("\n🔄 ALGORITHMIC FLOW:")
    print("1. Iterate through historical data minute by minute")
    print("2. Identify Peak_Price within Lookback_Period_For_Peak")
    print("3. Detect Drop Event when price falls below threshold")
    print("4. Ensure drop sustains for Minimum_Drop_Event_Duration") 
    print("5. Identify Trough_Price as lowest point after drop")
    print("6. Calculate Target_Recovery_Price")
    print("7. Monitor subsequent intervals for recovery")
    print("8. Record as Successful Recovery Event or failed event")
    print("="*80)
    print()
    
    # Display current configuration from main function
    from inspect import getsource
    print("Loading configuration...")
    print()
    
    results = main()
    
    if results is not None and not results.empty:
        print("\nAnalysis completed successfully!")
        print(f"Total events analyzed: {results['Total Drop Events Observed'].sum()}")
    else:
        print("\nAnalysis failed - no results generated")