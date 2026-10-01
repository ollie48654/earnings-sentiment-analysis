import os
import json
import time
import pandas as pd
import yfinance as yf
from pathlib import Path
from datetime import datetime, timedelta
from datasets import load_dataset

DATA_DIR = Path(__file__).parent.parent / 'data'
TRANS_DIR = DATA_DIR / 'transcripts'
PROC_DIR  = DATA_DIR / 'processed'
TRANS_DIR.mkdir(parents=True, exist_ok=True)
PROC_DIR.mkdir(parents=True, exist_ok=True)

# 20 major companies with active earnings calls
COMPANIES = {
    'AAPL': 'Apple',      'MSFT': 'Microsoft',  'GOOGL': 'Alphabet',
    'AMZN': 'Amazon',     'NVDA': 'NVIDIA',      'META': 'Meta',
    'TSLA': 'Tesla',      'JPM': 'JPMorgan',     'GS': 'Goldman Sachs',
    'NFLX': 'Netflix',    'AMD': 'AMD',           'INTC': 'Intel',
    'CRM': 'Salesforce',  'ORCL': 'Oracle',      'IBM': 'IBM',
    'BABA': 'Alibaba',    'UBER': 'Uber',         'SNAP': 'Snap',
    'TWTR': 'Twitter',    'SHOP': 'Shopify'
}

def fetch_transcripts():
    print("Loading earnings call dataset from Hugging Face...")
    try:
        dataset = load_dataset("lamini/earnings-calls-qa", split="train")
        df = dataset.to_pandas()
        print(f"Loaded {len(df):,} records")
        print(f"Columns: {list(df.columns)}")
        df.to_csv(PROC_DIR / 'raw_transcripts.csv', index=False)
        return df
    except Exception as e:
        print(f"Hugging Face dataset failed: {e}")
        print("Falling back to manual transcript creation...")
        return create_sample_transcripts()

def create_sample_transcripts():
    """
    Creates a realistic dataset of earnings call excerpts
    if the Hugging Face dataset is unavailable.
    Each entry represents a paragraph from an earnings call.
    """
    transcripts = [
        {"ticker":"AAPL","company":"Apple","quarter":"Q3 2023","date":"2023-08-03",
         "text":"We are thrilled to report record revenue this quarter driven by exceptional iPhone and services performance. Our installed base has never been stronger and customer satisfaction remains at all-time highs. We see tremendous opportunity ahead."},
        {"ticker":"AAPL","company":"Apple","quarter":"Q3 2023","date":"2023-08-03",
         "text":"Mac revenue declined year over year reflecting a challenging macroeconomic environment and tough comparisons against the prior year period which benefited from strong pent-up demand."},
        {"ticker":"MSFT","company":"Microsoft","quarter":"Q4 2023","date":"2023-07-25",
         "text":"Azure and other cloud services revenue grew 26 percent year over year. AI is absolutely driving new workloads and we are seeing strong customer demand across every segment of our business."},
        {"ticker":"MSFT","company":"Microsoft","quarter":"Q4 2023","date":"2023-07-25",
         "text":"We continue to invest heavily in our AI infrastructure which will pressure margins in the near term but we are confident these investments will generate substantial returns over the long run."},
        {"ticker":"GOOGL","company":"Alphabet","quarter":"Q2 2023","date":"2023-07-25",
         "text":"Search and other revenues grew 5 percent reflecting the value we provide to users and advertisers. YouTube ads showed recovery and we are excited about the momentum in Google Cloud which grew 28 percent."},
        {"ticker":"GOOGL","company":"Alphabet","quarter":"Q2 2023","date":"2023-07-25",
         "text":"We faced headwinds from foreign exchange and the ongoing challenging advertising market but our disciplined approach to expenses resulted in meaningful margin improvement."},
        {"ticker":"AMZN","company":"Amazon","quarter":"Q2 2023","date":"2023-08-03",
         "text":"AWS delivered strong results with revenue of 23.8 billion dollars growing 12 percent year over year. We are seeing early signs of cost optimisation headwinds stabilising and new workload growth accelerating."},
        {"ticker":"NVDA","company":"NVIDIA","quarter":"Q2 2024","date":"2023-08-23",
         "text":"Data center revenue was a record 10.3 billion dollars up 171 percent from a year ago. Demand for our H100 GPUs is extraordinary and exceeds our ability to supply. The generative AI era has arrived."},
        {"ticker":"NVDA","company":"NVIDIA","quarter":"Q2 2024","date":"2023-08-23",
         "text":"We expect continued strong demand throughout the rest of the fiscal year. Our supply chain partners are working hard to increase capacity and we are confident in our ability to meet growing customer needs."},
        {"ticker":"META","company":"Meta","quarter":"Q2 2023","date":"2023-07-26",
         "text":"We had a good quarter driven by strong advertising revenue growth and continued progress on our efficiency efforts. Daily active people grew to 3.07 billion and monetisation metrics improved across all regions."},
        {"ticker":"META","company":"Meta","quarter":"Q2 2023","date":"2023-07-26",
         "text":"Reality Labs operating losses were 3.7 billion dollars in the quarter. We remain committed to our long-term vision for the metaverse but acknowledge these investments will continue to weigh on profitability for the foreseeable future."},
        {"ticker":"TSLA","company":"Tesla","quarter":"Q2 2023","date":"2023-07-19",
         "text":"We delivered 466 thousand vehicles in Q2 setting a new record. Our price reductions have stimulated demand and we believe we are well positioned for the long term despite near-term margin pressure."},
        {"ticker":"TSLA","company":"Tesla","quarter":"Q2 2023","date":"2023-07-19",
         "text":"Operating margin declined to 9.6 percent as vehicle average selling prices fell. We are aggressively cutting costs and believe the market for electric vehicles remains enormous."},
        {"ticker":"JPM","company":"JPMorgan","quarter":"Q2 2023","date":"2023-07-14",
         "text":"We reported record net income of 14.5 billion dollars. Net interest income was exceptional reflecting the higher rate environment. Credit remains healthy with low delinquencies across our consumer and commercial portfolios."},
        {"ticker":"GS","company":"Goldman Sachs","quarter":"Q2 2023","date":"2023-07-19",
         "text":"Investment banking revenues declined significantly reflecting subdued deal activity as clients remain cautious about the macroeconomic environment. We are focused on our core strengths while managing through this challenging period."},
        {"ticker":"NFLX","company":"Netflix","quarter":"Q2 2023","date":"2023-07-19",
         "text":"We added 5.9 million paid subscribers in Q2 exceeding our guidance. Password sharing initiatives are working better than expected and we are seeing strong engagement with our ad-supported tier."},
        {"ticker":"AMD","company":"AMD","quarter":"Q2 2023","date":"2023-08-01",
         "text":"Data center GPU revenue more than doubled sequentially as AI demand accelerates. We are very excited about our MI300 roadmap and believe we are well positioned to gain share in the rapidly growing AI accelerator market."},
        {"ticker":"INTC","company":"Intel","quarter":"Q2 2023","date":"2023-07-27",
         "text":"Revenue was 12.9 billion dollars, above our guidance range. We are making progress on our process roadmap and remain on track for five nodes in four years. The PC market is showing early signs of recovery."},
        {"ticker":"INTC","company":"Intel","quarter":"Q2 2023","date":"2023-07-27",
         "text":"We continue to face significant competition from our rivals and the road ahead remains challenging. Our foundry ambitions require substantial capital investment and execution risk remains elevated."},
        {"ticker":"CRM","company":"Salesforce","quarter":"Q2 2024","date":"2023-08-30",
         "text":"Revenue grew 11 percent to 8.6 billion dollars and we raised full year guidance. Einstein AI is driving meaningful customer interest and we are seeing strong pipeline growth from our AI-powered products."},
        {"ticker":"UBER","company":"Uber","quarter":"Q2 2023","date":"2023-08-01",
         "text":"We achieved our first ever GAAP operating profit quarter. Gross bookings grew 16 percent to 33.6 billion dollars and both our mobility and delivery segments delivered strong results."},
        {"ticker":"SNAP","company":"Snap","quarter":"Q2 2023","date":"2023-07-25",
         "text":"Revenue declined 4 percent year over year as the digital advertising market remained soft. We are making difficult decisions to reduce costs and refocus our investments on our highest priority initiatives."},
        {"ticker":"SHOP","company":"Shopify","quarter":"Q2 2023","date":"2023-08-02",
         "text":"Revenue grew 31 percent to 1.7 billion dollars driven by strong merchant growth and expanding international presence. We completed the divestiture of our logistics business allowing us to focus on our core platform."},
    ]
    df = pd.DataFrame(transcripts)
    df.to_csv(PROC_DIR / 'raw_transcripts.csv', index=False)
    print(f"Created {len(df)} sample transcript entries")
    return df

def fetch_stock_returns(df):
    print("Fetching stock price data...")
    
    # Hardcoded clean dates for each company
    earnings_dates = {
        'AAPL': '2023-08-03', 'MSFT': '2023-07-25', 'GOOGL': '2023-07-25',
        'AMZN': '2023-08-03', 'NVDA': '2023-08-23', 'META': '2023-07-26',
        'TSLA': '2023-07-19', 'JPM': '2023-07-14', 'GS': '2023-07-19',
        'NFLX': '2023-07-19', 'AMD': '2023-08-01', 'INTC': '2023-07-27',
        'CRM': '2023-08-30', 'UBER': '2023-08-01', 'SNAP': '2023-07-25',
        'SHOP': '2023-08-02'
    }

    results = []
    for ticker, date_str in earnings_dates.items():
        try:
            date  = pd.to_datetime(date_str)
            start = date - timedelta(days=5)
            end   = date + timedelta(days=10)

            stock = yf.download(ticker, start=start, end=end,
                                progress=False, auto_adjust=True)
            if stock.empty:
                continue

            stock = stock.reset_index()
            stock.columns = [c[0] if isinstance(c, tuple) else c
                             for c in stock.columns]
            stock.columns = [c.lower() for c in stock.columns]
            stock['date'] = pd.to_datetime(stock['date'])

            on_day   = stock[stock['date'] >= date].head(1)
            after_3d = stock[stock['date'] >= date + timedelta(days=3)].head(1)

            if on_day.empty or after_3d.empty:
                continue

            price_day   = float(on_day['close'].iloc[0])
            price_after = float(after_3d['close'].iloc[0])
            ret_3d = round((price_after - price_day) / price_day * 100, 4)

            results.append({
                'ticker': ticker,
                'company': COMPANIES.get(ticker, ticker),
                'earnings_date': date_str,
                'price_on_day': round(price_day, 2),
                'price_3d_after': round(price_after, 2),
                'return_3d_pct': ret_3d
            })
            print(f"  {ticker}: {ret_3d:+.2f}%")
            time.sleep(0.5)

        except Exception as e:
            print(f"  {ticker}: failed — {e}")

    returns_df = pd.DataFrame(results)
    returns_df.to_csv(PROC_DIR / 'stock_returns.csv', index=False)
    print(f"Saved stock_returns.csv — {len(returns_df)} companies")
    return returns_df

if __name__ == '__main__':
    print("Starting...")
    df = create_sample_transcripts()
    print(f"Transcripts ready: {len(df)} rows")
    returns_df = fetch_stock_returns(df)
    print("Data collection complete.")