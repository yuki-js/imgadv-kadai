#!/usr/bin/env python3
"""
Removes assets listed after 2025-01-01 and merges L1/L2 classes into 'Native'.

- Input:  02_classify/output/assets.csv
- Output: 02_classify/output/assets_native_pre2025.csv

Keeps header. Compares 'listing_date' column (format: YYYY-MM-DD).
If 'class' is 'L1' or 'L2', set to 'Native'.
"""

import csv
from datetime import datetime

INPUT_CSV = "02_classify/output/assets.csv"
OUTPUT_CSV = "02_classify/output/assets_native_pre2025.csv"
CUTOFF_DATE = datetime(2025, 1, 1)

def main():
    with open(INPUT_CSV, newline='', encoding='utf-8') as fin, \
         open(OUTPUT_CSV, 'w', newline='', encoding='utf-8') as fout:
        reader = csv.DictReader(fin)
        writer = csv.DictWriter(fout, fieldnames=reader.fieldnames)
        writer.writeheader()
        for row in reader:
            date_str = row.get('listing_date', '')
            try:
                date_obj = datetime.strptime(date_str, "%Y-%m-%d")
            except Exception:
                continue
            if date_obj < CUTOFF_DATE:
                cls = row.get('class', '')
                if cls in ('L1', 'L2'):
                    row['class'] = 'Native'
                writer.writerow(row)

if __name__ == "__main__":
    main()