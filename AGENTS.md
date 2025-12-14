# Principal Component Analysis (PCA) of Time-ordered Data

## Dataset 

Create a new dataset, which holds cryptocurrency price data over time. 100 high volume cryptocurrencies are selected, and their daily closing prices over the past 3 years are collected. The dataset is structured such that each row represents a different cryptocurrency, and each column represents the closing price on a specific day.

## What you must achieve

- Implement PCA on the dataset to reduce its dimensionality while retaining as much variance as possible.
- Visualize the first 9 principal components as time series plots.
- Analyze and interpret the results, discussing how much variance is explained by the principal components and what insights can be drawn from the time series plots.

## Steps to do that 

1. Setup this python workspace by calling pip. We are already on venv so just run pip install xxx.
2. Collect the daily closing prices of the top 100 cryptocurrencies over the past 3 years.
3. Do PCA with some libs.
4. Notify me when PCA done

## Where to get data

Binance API is a good source for cryptocurrency price data. Coingecko is bad since it has rate limits and data access restrictions.

For top 100 cryptocurrencies by market cap, you can pick up at the point of 3 years ago. It's because some cryptos are new and don't have 3 years data.

## Notes

- Use multiple files for better readability.
- you must split the data collection phase, and other phase into different files.