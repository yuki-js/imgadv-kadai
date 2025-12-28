# Role
You are a data analysis assistant specializing in Pattern Recognition and Signal Processing. Your task is to generate code and execute an analysis for a university report titled "Advanced Image Recognition".

# Context: Assignment Requirements (Source Text from Slides)
The following is the exact text extracted from the lecture slides. You must strictly adhere to these requirements.

--- [Requirement 1: Tasks] ---
1. 4クラスの高次元ベクトルデータセットに重判別分析を適用して 判別空間へ射影したデータ分布トを3次元プロットせよ。
2. 1.で得られた判別ベクトルを画像として可視化せよ、
3. 4クラス全体に主成分分析を適用し その結果を3次元プロットせよ。
4. 3で得られた基底ベクトルを画像として可視化せよ。
5. 1~4に基づいて、重判別分析と主成分分析の結果を比較し、その違いを様々な観点から詳しく考察せよ。

--- [Requirement 2: Notes] ---
* 対象とするデータは、自分の研究に関連するデータを独自収集とすること。
* 各クラスの収集データ数は出来るだけ多くする.
* “ランク落ち“しないように注意して計算する.
* 実験・解析にはMatlab, R, Scilabなど用いる.

--- [Requirement 3: Deliverables] ---
* 以下の項目を含む(説明し易い構成でOK)
    * 評価データに関する説明(収集法,データ次元や特性など)
    * 3次元プロット図
    * 可視化(画像化)した第 1~3 基底ベクトル
    * 考察

# Instructions for the AI Agent
Please write and execute code (using Python, MATLAB, or the most appropriate library in the current environment) to perform the following steps.

**Step 0: Load Existing Project (PRIORITY)**
**CRITICAL:** Before generating new data or code, **check and load the current project files** (e.g., existing `.csv`, `.py`, `.m` files, or dataframes in memory).
* If a dataset or script already exists, use it and only modify the parts necessary to meet the visualization requirements below.
* Do not overwrite existing progress unless necessary.

**Step 1: Data Acquisition (Abstracted)**
If data is missing, acquire historical price data (Close price) for **4 distinct asset classes**.
* **Class Mapping (Target 4 Classes):**
    Use the following asset classes to represent the 4 classes. Examples are given in parentheses:
    1.  **L1:** `SOL` (Solana) `TRX` (Tron)
    2.  **L2/sidechains:** `POL` (Polygon) `ARB` (Arbitrum) `OP` (Optimism)
    3.  **Protocol:** `JUP` (Jupiter) `PUMP` (Pump.fun governance token, don't to be confused with tokens listed in Pump.fun)
    4.  **Meme:** `PEPE` (Pepe) `GOAT` (Goatseus Maximus) `TRUMP` (Official Trump Coin) `REALTRUMP` (Unofficial Trump Coin) 
* **Data Structure:**
    * Treat a time-series segment of length $D$ (approx. 100) as a single data vector.
    * *Quantity:* Ensure $N_{total} > D$ (approx. 50 samples per class) to satisfy the "Rank Deficiency" constraint mentioned in the slides.
    * All

**Step 2: Preprocessing**
* **Normalization:** Normalize each segment (e.g., percentage change) to focus on the **waveform shape**.
* **Labeling:** Assign labels corresponding to the 4 classes above.

**Step 3: Analysis & Visualization**
Generate the following outputs strictly required for the report:
* **A. LDA 3D Plot:** Scatter plot of the 4 classes (L1, L2, Protocol, Meme) in the discriminant space.
* **B. LDA Basis Visualization:** Plot the discriminant vectors as **Line Graphs** (interpreting "Image Visualization" for 1D data).
* **C. PCA 3D Plot:** Scatter plot of the first 3 Principal Components.
* **D. PCA Basis Visualization:** Plot the eigenvectors (PC1, PC2, PC3) as **Line Graphs**.

# Output Requirements
* Provide the executable code.
* Save the plots as high-resolution image files.
# Dataset


Create a new dataset, which holds cryptocurrency price data over time. 100 high volume cryptocurrencies are selected, and their daily closing prices over the past 3 years are collected. The dataset is structured such that each row represents a different cryptocurrency, and each column represents the closing price on a specific day.

## Where to get data

Binance API is a good source for cryptocurrency price data. Coingecko is bad since it has rate limits and data access restrictions.

For top 100 cryptocurrencies by market cap, you can pick up at the point of 3 years ago. It's because some cryptos are new and don't have 3 years data.

## Notes

- Use multiple files for better readability.
- you must split the data collection phase, and other phase into different files.