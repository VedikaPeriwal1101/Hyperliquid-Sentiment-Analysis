import pandas as pd
import numpy as np


#Loading Dataset into memory
df_sentiment = pd.read_csv("C:/Users/Vedika/Downloads/fear_greed_index.csv")
df_trades = pd.read_csv("C:/Users/Vedika/Downloads/historical_data.csv")

print("Status: Data loaded into the memory")

#Checking for null data

print ("---Missing Data Audit (fear_greed_index)---")
print(df_sentiment.isnull().sum())

print ("---Missing Data Audit(historical_data)---")
print(df_trades.isnull().sum())

# 1. Force the exact format: %d (Day) - %m (Month) - %Y (4-digit Year) %H:%M (Time)
df_trades['Date_Key'] = pd.to_datetime(df_trades['Timestamp IST'], format='%d-%m-%Y %H:%M', errors='coerce').dt.date

# 2. Format the Sentiment Date (Assuming it's a standard format, dayfirst=True acts as a safety net)
df_sentiment['Date_Key'] = pd.to_datetime(df_sentiment['date'], dayfirst=True, errors='coerce').dt.date

# 3. Execute the Merge
df_merged = pd.merge(df_trades, df_sentiment, on='Date_Key', how='left')


print("Trades Start Date: ", df_trades['Date_Key'].min())
print("Trades End Date:   ", df_trades['Date_Key'].max())
print("Sentiment Start Date:", df_sentiment['Date_Key'].min())
print("Sentiment End Date:  ", df_sentiment['Date_Key'].max())


# 1. Check the Time Horizons (Ignoring the garbage NaNs)
print("--- TIME HORIZON AUDIT ---")
print("Trades Start Date: ", df_trades['Date_Key'].dropna().min())
print("Trades End Date:   ", df_trades['Date_Key'].dropna().max())
print("Sentiment Start Date:", df_sentiment['Date_Key'].dropna().min())
print("Sentiment End Date:  ", df_sentiment['Date_Key'].dropna().max())

# 2. Count the Garbage
print("\n--- CORRUPTION AUDIT ---")
print("Corrupted/Missing Dates in Trades:   ", df_trades['Date_Key'].isnull().sum())
print("Corrupted/Missing Dates in Sentiment:", df_sentiment['Date_Key'].isnull().sum())

# 1. Hardcode the Trades parser (DD-MM-YYYY HH:MM)
df_trades['Date_Key'] = pd.to_datetime(df_trades['Timestamp IST'], format='%d-%m-%Y %H:%M', errors='coerce').dt.date

# 2. Hardcode the Sentiment parser (YYYY-MM-DD)
df_sentiment['Date_Key'] = pd.to_datetime(df_sentiment['date'], format='%Y-%m-%d', errors='coerce').dt.date

# 3. Execute the Inner Join (Only keep trades with matching sentiment)
df_final = pd.merge(df_trades, df_sentiment, on='Date_Key', how='inner')

# 4. The Final Audit
print("--- FINAL RECOVERY AUDIT ---")
print("Recovered Trades for Analysis:", df_final.shape[0])
print("Missing Sentiment Values:   ", df_final['classification'].isnull().sum())


# 4. The Audit
print("Merged Matrix Dimensions:", df_merged.shape)
print("Unmapped Sentiment (Nulls):", df_merged['classification'].isnull().sum())

# 1. Purge the 21 unmapped sentiment rows
df_merged = df_merged.dropna(subset=['classification'])

# 2. Force 'Closed PnL' to numeric (safeguard against hidden string characters)
df_merged['Closed PnL'] = pd.to_numeric(df_merged['Closed PnL'], errors='coerce').fillna(0)

# 3. Engineer the Trader Profiling Matrix
trader_profiles = df_merged.groupby('Account').agg(
    Total_Trades=('Closed PnL', 'count'),
    Total_PnL=('Closed PnL', 'sum'),
    Winning_Trades=('Closed PnL', lambda x: (x > 0).sum())
)

# 4. Calculate the Win Rate Percentage
trader_profiles['Win_Rate_%'] = (trader_profiles['Winning_Trades'] / trader_profiles['Total_Trades']) * 100

# 5. The Audit (Identify our top 5 apex traders)
print("Apex Traders by Profitability:")

# 1. Group the entire merged dataset by the Fear/Greed classification
sentiment_matrix = df_merged.groupby('classification').agg(
    Total_Trades=('Closed PnL', 'count'),
    Total_PnL=('Closed PnL', 'sum'),
    Average_PnL_Per_Trade=('Closed PnL', 'mean'),
    Winning_Trades=('Closed PnL', lambda x: (x > 0).sum())
)

# 2. Calculate the global Win Rate across sentiment zones
sentiment_matrix['Win_Rate_%'] = (sentiment_matrix['Winning_Trades'] / sentiment_matrix['Total_Trades']) * 100

# 3. Force Pandas to show the money (formatting the output to prevent truncation)
pd.set_option('display.float_format', lambda x: '%.2f' % x)
print(sentiment_matrix[['Total_Trades', 'Total_PnL', 'Average_PnL_Per_Trade', 'Win_Rate_%']])



# 1. Ensure absolute numeric safety for financial columns
df_final['Size USD'] = pd.to_numeric(df_final['Size USD'], errors='coerce').fillna(0)
df_final['Closed PnL'] = pd.to_numeric(df_final['Closed PnL'], errors='coerce').fillna(0)

# 2. Handle Leverage (Checks if 'leverage' or 'Leverage' exists, defaults to 1 if missing)
lev_col = 'Leverage' if 'Leverage' in df_final.columns else ('leverage' if 'leverage' in df_final.columns else None)
if lev_col:
    df_final[lev_col] = pd.to_numeric(df_final[lev_col], errors='coerce').fillna(1)

# 3. Build the Trader Feature Matrix
trader_metrics = df_final.groupby('Account').agg(
    Total_Trades=('Trade ID', 'count'),
    Total_PnL=('Closed PnL', 'sum'),
    Win_Rate_Pct=('Closed PnL', lambda x: (x > 0).mean() * 100),
    Avg_Trade_Size_USD=('Size USD', 'mean'),
    # Count how many trades were Long vs Short (using 'Side' or 'Direction')
    Total_Longs=('Side', lambda x: x.astype(str).str.contains('Buy|Long', case=False).sum()),
    Total_Shorts=('Side', lambda x: x.astype(str).str.contains('Sell|Short', case=False).sum())
)

# 4. Engineer Advanced Ratios
# Long/Short Ratio (Adding 1 to denominator to prevent division by zero)
trader_metrics['Long_Short_Ratio'] = trader_metrics['Total_Longs'] / (trader_metrics['Total_Shorts'] + 1)

# Add average leverage if the column existed
if lev_col:
    trader_metrics['Avg_Leverage'] = df_final.groupby('Account')[lev_col].mean()

# 5. Filter out the noise (Only keep traders with more than 5 trades to avoid skewed win rates)
trader_metrics = trader_metrics[trader_metrics['Total_Trades'] > 5]

pd.set_option('display.float_format', lambda x: '%.2f' % x)
print("--- TRADER FEATURE MATRIX BUILT ---")
print(f"Total Unique Valid Traders: {trader_metrics.shape[0]}")
print(trader_metrics.head())

# 1. Ensure numeric columns
df_final['Size USD'] = pd.to_numeric(df_final['Size USD'], errors='coerce').fillna(0)
df_final['Closed PnL'] = pd.to_numeric(df_final['Closed PnL'], errors='coerce').fillna(0)

# 2. Leverage safety check
lev_col = 'Leverage' if 'Leverage' in df_final.columns else ('leverage' if 'leverage' in df_final.columns else None)
if lev_col:
    df_final[lev_col] = pd.to_numeric(df_final[lev_col], errors='coerce').fillna(1)

# 3. Build the Master Sentiment Matrix
sentiment_behavior = df_final.groupby('classification').agg(
    Total_Trades=('Closed PnL', 'count'),
    Total_PnL=('Closed PnL', 'sum'),
    Win_Rate_Pct=('Closed PnL', lambda x: (x > 0).mean() * 100),
    Avg_Loss_Proxy=('Closed PnL', lambda x: x[x < 0].mean()), # Proxy for drawdown
    Avg_Trade_Size_USD=('Size USD', 'mean'),
    Total_Longs=('Side', lambda x: x.astype(str).str.contains('Buy|Long', case=False).sum()),
    Total_Shorts=('Side', lambda x: x.astype(str).str.contains('Sell|Short', case=False).sum())
)

# Add Leverage and L/S Ratio
if lev_col:
    sentiment_behavior['Avg_Leverage'] = df_final.groupby('classification')[lev_col].mean()
sentiment_behavior['L_S_Ratio'] = sentiment_behavior['Total_Longs'] / (sentiment_behavior['Total_Shorts'] + 1)

# Clean up output
columns_to_show = ['Total_Trades', 'Total_PnL', 'Win_Rate_Pct', 'Avg_Loss_Proxy', 'Avg_Trade_Size_USD', 'L_S_Ratio']
if lev_col: columns_to_show.append('Avg_Leverage')

pd.set_option('display.float_format', lambda x: '%.2f' % x)
print("--- BEHAVIOR & PERFORMANCE BY SENTIMENT ---")
print(sentiment_behavior[columns_to_show])

# 1. Calculate per-account baseline metrics (Added .reset_index() to fix the error)
trader_segments = df_final.groupby('Account').agg(
    Total_Trades=('Closed PnL', 'count'),
    Total_PnL=('Closed PnL', 'sum'),
    Win_Rate=('Closed PnL', lambda x: (x > 0).mean() * 100)
).reset_index()

# 2. Define the Segments
# Segment A: High Frequency vs Low Frequency
median_trades = trader_segments['Total_Trades'].median()
trader_segments['Frequency_Segment'] = np.where(trader_segments['Total_Trades'] >= median_trades, 'High Frequency', 'Low Frequency')

# Segment B: Consistent Winners vs Inconsistent
trader_segments['Consistency_Segment'] = np.where(
    (trader_segments['Win_Rate'] > 50) & (trader_segments['Total_PnL'] > 0), 
    'Consistent Winner', 
    'Inconsistent/Loser'
)

# 3. Print the Segmentation Tables
pd.set_option('display.float_format', lambda x: '%.2f' % x)
print("\n--- SEGMENT 1: FREQUENCY PROFILES ---")
print(trader_segments.groupby('Frequency_Segment').agg({'Total_PnL':'sum', 'Win_Rate':'mean', 'Total_Trades':'mean'}))

print("\n--- SEGMENT 2: CONSISTENCY PROFILES ---")
print(trader_segments.groupby('Consistency_Segment').agg({'Total_PnL':'sum', 'Total_Trades':'mean', 'Account':'size'}).rename(columns={'Account':'Trader_Count'}))



# 1. Export the Market Sentiment Table
sentiment_behavior.to_csv('1_Sentiment_Behavior.csv')

# 2. Export the Frequency Segments Table
freq_table = trader_segments.groupby('Frequency_Segment').agg({'Total_PnL':'sum', 'Win_Rate':'mean', 'Total_Trades':'mean'})
freq_table.to_csv('2_Frequency_Segments.csv')

# 3. Export the Consistency Segments Table
cons_table = trader_segments.groupby('Consistency_Segment').agg({'Total_PnL':'sum', 'Total_Trades':'mean', 'Account':'size'}).rename(columns={'Account':'Trader_Count'})
cons_table.to_csv('3_Consistency_Segments.csv')

print("SUCCESS: 3 CSV files have been created in your working directory.")



