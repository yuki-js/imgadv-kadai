# assets.csv Specification (v1.0.0)

Status: Stable  
Owner: Data Engineering / Quant Research  
Applies to: Dataset master of crypto assets used for time-series analysis

---

1. Purpose and Scope

This document specifies the exact structure, encoding, constraints, and validation logic for the CSV file that defines the asset universe for downstream data collection and analysis. Implementers must adhere strictly to this specification to ensure:
- Consistency across automated data collection (Binance API)
- Correct class mapping into 5 fixed classes (L1, L2, Protocol, Meme, GameFi)
- Proper handling of exchange-specific symbol differences
- Explicit tracking of out-of-class and excluded assets

This spec applies to the CSV located at:
- data path: data/assets.csv
- content: 100 included assets (exclude == False) that belong to the 5 fixed classes, plus any number of excluded assets used for documentation and audit purposes

---

2. File Location and Format

- Location: data/assets.csv
- Encoding: UTF-8 (no BOM)
- Line endings: LF (\n) preferred
- Delimiter: comma (,)
- Quote char: " (double quote) — fields containing commas must be quoted
- Header row: required (single header row listing column names exactly as specified below)
- Row order: deterministic and stable across commits (recommended: sort by class, then symbol_common)

---

3. Row Semantics

- Each row represents exactly one token (base asset) to be considered by the dataset.
- A token is identified for Binance API collection by symbol_binance (trading pair string such as BTCUSDT).
- Class assignment is fixed to one of the five classes (L1, L2, Protocol, Meme, GameFi). Out-of-class assets must use an empty class ("") and be marked exclude=True with an appropriate exclude_reason.
- Exactly 100 assets must have exclude=False and non-empty class among the five fixed classes. Excluded rows are allowed but do not count toward this 100.

---

4. Column Definitions (Required Columns)

The CSV must contain the following columns in this exact order:

1) symbol_binance
- Type: string (non-empty)
- Description: Binance trading pair used for historical data collection (e.g., BTCUSDT, ETHUSDT).
- Constraints:
  - Must correspond to a valid Binance spot symbol that returns daily klines (interval=1d).
  - Quote asset must be USDT whenever available. If a USDT market does not exist for a token, the asset should be excluded (exclude=True, exclude_reason="no_usdt_pair"), unless explicitly approved to use a different quote.
  - Uniqueness: symbol_binance must be unique across the file.

2) symbol_common
- Type: string (non-empty)
- Description: Common or official ticker symbol used for humans (e.g., BTC, ETH, ARB, CRV, STRK).
- Notes:
  - This can differ from the on-chain symbol or other exchanges.
  - Used for human-readable references and chart labels.

3) class
- Type: enum string
- Allowed values: "L1", "L2", "Protocol", "Meme", "GameFi", or "" (empty string).
- Description:
  - Five-class system: L1, L2, Protocol, Meme, GameFi.
  - Use empty string "" only for out-of-class assets that must be excluded (exclude=True).
- Rules:
  - For exclude == False: class must be one of {L1, L2, Protocol, Meme, GameFi}.
  - For exclude == True: class must be "".

4) name
- Type: string (non-empty)
- Description: Official token name (e.g., Bitcoin, Ethereum, StarkNet, Curve).

5) listing_date
- Type: date string (YYYY-MM-DD)
- Description: Binance listing date (or the effective data start date if exact listing is unavailable).
- Usage:
  - Used to enforce the global dataset collection window. The effective data start for a token is max(listing_date, dataset_start_date).
  - If listing_date is within the last 3 years, collection must begin at listing_date.

6) description
- Type: string (non-empty)
- Description: Token-centric description focusing on the token’s role, utility, and economic characteristics, not the project’s product.
- Style guide:
  - Emphasize token functions (e.g., governance, native currency, staking collateral, fee token).
  - Include salient monetary properties if relevant (e.g., capped supply, inflationary, staking rewards).
  - Keep one or two sentences max; concise and unambiguous.

7) exclude
- Type: boolean literal ("True" / "False")
- Description: Whether this token is excluded from the dataset.
- Rules:
  - Exclude out-of-class assets, wrapped assets, stablecoins, exchange utility tokens, rebase tokens, synthetic/bridged assets, delisted or illiquid assets.
  - The 100-asset universe must all have exclude == False.

8) exclude_reason
- Type: string (allow empty)
- Description: Reason code for exclusion. Required when exclude == True, otherwise must be empty.
- Allowed codes (controlled vocabulary):
  - "stablecoin" (e.g., USDT, USDC, DAI)
  - "wrapped" (e.g., WBTC, WETH)
  - "utility" (extreme utility tokens not appropriate for class analytics)
  - "exchange_token" (exchange/platform utility tokens)
  - "rebase" (tokens with supply rebasing mechanics)
  - "synthetic" (synthetic or bridged representations)
  - "no_usdt_pair" (no Binance USDT market)
  - "low_liquidity" (insufficient liquidity/volume)
  - "delisted" (no longer actively traded)
  - "duplicate" (duplicate or conflicting entry)
  - "unknown" (temporary placeholder; must be resolved before freeze)

9) classification_note
- Type: string (allow empty)
- Description: Free-form notes capturing classification rationale or manual overrides (e.g., "Governance token but functionally L2; assigned to L2 by rule X").

---

5. Validation Rules

5.1 CSV-Level Rules
- The header must include exactly the 9 columns in the defined order.
- The file must be UTF-8 without BOM.
- Exactly 100 rows must have exclude == False and class in {L1, L2, Protocol, Meme, GameFi}.
- No duplicate symbol_binance values.

5.2 Row-Level Rules
- symbol_binance: must be a Binance spot pair string and pass a kline dry-run test (HTTP 200 and data rows) under /api/v3/klines with interval=1d.
- symbol_common: non-empty.
- class: if exclude == False, must be one of the 5 classes; if exclude == True, must be "".
- name: non-empty.
- listing_date: YYYY-MM-DD, not in the future relative to data collection time.
- description: non-empty; must follow token-centric style.
- exclude: "True" or "False".
- exclude_reason: required (non-empty) if exclude == True; empty if exclude == False.
- classification_note: optional; can be empty.
- Prohibited: inferring class or exclusion from ticker strings (e.g., “W” prefix), “well-known” code names, or substrings; rely only on authoritative metadata (Binance product endpoints, issuer documentation), curated taxonomy, or documented manual review.

5.3 Dataset Window and listing_date
- Global dataset window: last 3 years by default.
- Effective start for each asset = max(listing_date, global dataset start).
- If effective start reduces available samples below requirements, the asset should be replaced or the sample strategy adapted (e.g., more assets per class) to preserve N_total > D.

---

6. Class Assignment Policy

- The five classes are fixed: L1, L2, Protocol, Meme, GameFi.
- Decision rules (rule-based first, manual confirmation second):
  - L1: Native token of a base layer blockchain used for fees/security (e.g., BTC, ETH, SOL).
  - L2: Governance/utility token for a Layer 2/sidechain/rollup (e.g., ARB, OP, STRK).
  - Protocol: Governance/utility token of a DeFi or on-chain service protocol (e.g., UNI, AAVE, CRV).
  - Meme: Community/meme-driven tokens with limited intrinsic utility (e.g., DOGE, SHIB, PEPE).
  - GameFi: Tokens used for blockchain gaming ecosystems (e.g., AXS, SAND, GALA).
- Out-of-class assets (e.g., stablecoins, wrapped, exchange tokens) must be recorded with class="" and exclude=True with a proper exclude_reason.
- When uncertain: set class="" and exclude=True with exclude_reason="unknown" and provide a classification_note; such rows do not count toward the 100 and must be resolved before finalization.

---

7. Description Authoring Guidelines (Token-Centric)

- Focus on the token’s role:
  - Native fee token, governance rights, staking utility, collateralization.
- Economic properties:
  - Supply model (fixed, inflationary), staking rewards, revenue share (if any).
- Do not describe the full product or protocol at length; restrict to 1–2 sentences about the token.
- Examples:
  - BTC: "Native currency of the Bitcoin network used as a decentralized medium of exchange and store of value; capped supply of 21 million and secured by PoW mining."
  - ARB: "Governance token for the Arbitrum ecosystem enabling DAO voting on protocol parameters and treasury; utility may expand to fee mechanisms over time."

---

8. Exchange Symbol Differences

- The CSV disambiguates symbols via:
  - symbol_binance: Binance trading pair string for data collection (e.g., STRKUSDT, CRVUSDT).
  - symbol_common: A human-readable ticker symbol used in reports (e.g., STRK, CRV).
- Never derive binance symbols from symbol_common by concatenation; always verify via Binance API or whitelisted mapping.
- Use USDT quote by default. If no USDT market exists, set exclude=True with exclude_reason="no_usdt_pair" unless explicitly overridden.
- Important: Do not use ticker-code string heuristics for classification or exclusion.
  - Ticker code structure is not a reliable semantic signal; “well-known” tickers may be renamed and vary across exchanges, leading to misclassification.
  - A leading “W” is not a dependable indicator of wrapped assets; many unrelated tokens begin with “W”.
  - Do not infer “wrapped”, “stablecoin”, “utility”, “GameFi”, or any class/exclusion from prefixes, suffixes, or substrings.
  - All class/exclusion decisions must be based on curated taxonomy, Binance product metadata/endpoints, or manual review recorded in classification_note.
  - symbol_binance is an API key only and must not be used as a signal for class/exclusion.

---

9. Exclusion Policy

- Exclude assets in the following categories:
  - Stablecoins (e.g., USDT, USDC, DAI)
  - Wrapped assets (e.g., WBTC, WETH)
  - Exchange utility tokens (e.g., used primarily for fee discounts)
  - Synthetic or bridged representations of other assets
  - Rebase tokens with supply adjustments that distort return series
  - Illiquid or delisted assets
  - Ambiguous assets pending review (temporarily mark as unknown)
- Heuristic prohibition: Do not use ticker strings (e.g., “W” prefix), “well-known” code names, or substring patterns to determine “wrapped”, “stablecoin”, or any exclusion/class status. Such statuses must be confirmed via authoritative metadata (e.g., Binance asset/product info, issuer documentation) or manual review recorded in classification_note.
- For all excluded assets:
  - class must be "", exclude must be True, and exclude_reason must be one of the allowed codes.

---

10. Examples

10.1 Example Rows (abbreviated)

| symbol_binance | symbol_common | class   | name        | listing_date | description                                                                                                                                                   | exclude | exclude_reason | classification_note                |
|----------------|---------------|---------|-------------|--------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------|---------|----------------|------------------------------------|
| BTCUSDT        | BTC           | L1      | Bitcoin     | 2010-07-17   | Native currency of the Bitcoin network used as a decentralized medium of exchange and store of value; capped at 21M; secured by Proof of Work.               | False   |                | L1 native token                    |
| ETHUSDT        | ETH           | L1      | Ethereum    | 2015-07-30   | Native token of Ethereum for gas and staking; used to pay transaction fees and secure the network via Proof of Stake; widely used as DeFi collateral.        | False   |                | L1 native token                    |
| SOLUSDT        | SOL           | L1      | Solana      | 2020-04-10   | Native token of Solana used for transaction fees and validator staking; inflationary emissions reward validators and secure high-throughput consensus.        | False   |                | L1 native token                    |
| ARBUSDT        | ARB           | L2      | Arbitrum    | 2023-03-23   | Governance token for the Arbitrum ecosystem enabling DAO voting on protocol upgrades and treasury; potential future fee utility under DAO governance.         | False   |                | L2 governance token                |
| OPUSDT         | OP            | L2      | Optimism    | 2022-04-27   | Governance/utility token for the Optimism network and OP Stack; used for protocol governance and incentives; may contribute to sequencer economics.           | False   |                | L2 governance/utility              |
| STRKUSDT       | STRK          | L2      | StarkNet    | 2024-02-20   | Governance/utility token for StarkNet; used to participate in protocol governance and potentially fee economics; associated with STARK-based L2 scaling.     | False   |                | L2 governance                      |
| CRVUSDT        | CRV           | Protocol| Curve       | 2020-08-13   | Governance token for Curve enabling DAO voting on AMM parameters and emissions; grants governance rights over stablecoin-focused liquidity pools.             | False   |                | DeFi governance                    |
| UNIUSDT        | UNI           | Protocol| Uniswap     | 2020-09-17   | Governance token of Uniswap protocol; token holders vote on protocol parameters and treasury allocations; governance-driven value accrual model.             | False   |                | DeFi governance                    |
| DOGEUSDT       | DOGE          | Meme    | Dogecoin    | 2013-12-06   | Inflationary meme currency primarily used for peer-to-peer transfers and tipping; value driven by community demand and sentiment rather than protocol utility.| False   |                | Meme/community                     |
| AXSUSDT        | AXS           | GameFi  | Axie Infinity| 2018-11-08  | Governance and utility token for Axie Infinity; used for governance voting and in-game economy functions such as breeding; earned via play-to-earn mechanics. | False   |                | GameFi governance/utility          |
| USDTUSDT       | USDT          |         | Tether      | 2017-10-11   | USD-pegged stablecoin used as settlement and liquidity across exchanges; centralized collateralized issuance; not an analytical asset-class token.            | True    | stablecoin     | Excluded stablecoin                |

Note: USDTUSDT is shown only for illustration of exclusion; in practice, stablecoins should be listed only if necessary for explicit exclusion tracking.

---

11. Implementation Notes

11.1 Reading with pandas
- Use dtype mappings to prevent unintended type inference:
  - symbol_binance: string
  - symbol_common: string
  - class: string
  - name: string
  - listing_date: string (parse later to datetime)
  - description: string
  - exclude: boolean (cast from "True"/"False")
  - exclude_reason: string
  - classification_note: string

11.2 API usage contract (Binance)
- For each row with exclude==False and class in {L1, L2, Protocol, Meme, GameFi}, request:
  - GET /api/v3/klines?symbol={symbol_binance}&interval=1d&startTime={ts_ms}&endTime={ts_ms}
- Ensure the trading pair exists and returns data.
- Use only USDT-quoted pairs, unless explicitly approved.

11.3 Listing date in pipeline
- dataset_start_date = today - 3 years (UTC, 00:00)
- effective_start_date = max(listing_date, dataset_start_date)
- If the effective_start_date shortens the available window so that N_total ≤ D for a given class, add more assets in that class or adjust segmentation to maintain the "no rank deficiency" requirement (N_total > D).

---

12. Quality Gates and Checks

- QG-1: Header and column order are exactly as specified.
- QG-2: Exactly 100 rows with exclude==False and valid class.
- QG-3: All included rows have non-empty description following token-centric style.
- QG-4: symbol_binance is unique and kline endpoint returns valid data for each included asset.
- QG-5: Excluded assets have class=="" and a non-empty exclude_reason from the allowed set.
- QG-6: listing_date is <= data collection end and in YYYY-MM-DD.
- QG-7: No stablecoins or wrapped assets are included (exclude must be True for those).
- QG-8: Deterministic ordering of rows.

---

13. Change Management

- Version the CSV via Git commits; do not reorder rows arbitrarily.
- Any change that affects the included set (exclude toggles, class changes) requires:
  - Update to this spec’s version in the header if rules change.
  - A commit message describing the rationale and the impact on class counts.
  - Re-running quality gates.

---

14. FAQ

Q: What if Binance’s human-readable symbol differs from common usage (e.g., STRK vs. STRKUSDT)?  
A: Always use symbol_binance for API collection (trading pair string) and symbol_common for human references. Never infer symbol_binance from symbol_common.

Q: How are ambiguous tokens handled?  
A: Set class="" and exclude=True with exclude_reason="unknown" and an explanatory classification_note; these rows are not part of the 100 included.

Q: Are stablecoins ever included?  
A: No. Stablecoins are tracked only as excluded rows if needed for documentation.

---

15. Appendix: Machine-Readable Row Schema (Informal JSON Schema)

This schema represents a single row (post-CSV parsing):

{
  "type": "object",
  "properties": {
    "symbol_binance": { "type": "string", "minLength": 1 },
    "symbol_common":  { "type": "string", "minLength": 1 },
    "class":          { "type": "string", "enum": ["L1", "L2", "Protocol", "Meme", "GameFi", ""] },
    "name":           { "type": "string", "minLength": 1 },
    "listing_date":   { "type": "string", "pattern": "^\\d{4}-\\d{2}-\\d{2}$" },
    "description":    { "type": "string", "minLength": 1 },
    "exclude":        { "type": "boolean" },
    "exclude_reason": { "type": "string" },
    "classification_note": { "type": "string" }
  },
  "required": [
    "symbol_binance", "symbol_common", "class", "name",
    "listing_date", "description", "exclude", "exclude_reason", "classification_note"
  ],
  "allOf": [
    {
      "if": { "properties": { "exclude": { "const": true } } },
      "then": {
        "properties": { "class": { "const": "" } },
        "required": ["exclude_reason"]
      }
    },
    {
      "if": { "properties": { "exclude": { "const": false } } },
      "then": {
        "properties": { "class": { "enum": ["L1", "L2", "Protocol", "Meme", "GameFi"] } },
        "not": { "properties": { "exclude_reason": { "minLength": 1 } } }
      }
    }
  ]
}
