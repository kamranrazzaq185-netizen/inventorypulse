# InventoryPulse — Stockout Risk Detection System

**Supply chain teams lose revenue to stockouts that were 
predictable days in advance. InventoryPulse uses sales 
velocity modeling to flag at-risk inventory before it 
becomes a crisis.**

---

## The Problem

Most supply chain dashboards report what already happened. 
By the time a stockout appears in a report, the sale is 
already lost and the customer is already frustrated.

The data to prevent this exists in every order management 
system. The signal isn't missing — it isn't being read correctly.

InventoryPulse answers one question:

> **Which products are going to stock out — and how many days do we have left?**

---

## Results

Analysis performed on 180,519 real supply chain orders 
across 3 years, 50 categories, and 118 unique SKUs.

| Risk Level | SKUs | Percentage |
|------------|------|------------|
| 🔴 Critical (0–7 days) | 9 | 7.6% |
| 🟠 At Risk (7–14 days) | 12 | 10.2% |
| 🟢 Safe (14+ days) | 97 | 82.2% |

**Key finding:** 8 products were already at zero stock 
with no active alert triggered. 17.8% of all SKUs required 
immediate inventory action.

Highest velocity at-risk product: Nike Men's CJ Elite 2 
TD Football Cleat — selling 17.6 units per day with 
only 12.2 days of stock remaining.

---

## Dashboard

<img width="1536" height="1024" alt="ChatGPT Image May 14, 2026, 04_44_14 PM" src="https://github.com/user-attachments/assets/e0a1ab68-fbff-4828-a9ba-c91aab036044" />


Three Important visuals. Three questions answered:
- Which products are critical right now?
- Which categories carry the most risk?
- How much runway does each SKU have left?

---

## How It Works

InventoryPulse uses a three-step sales velocity model:

**Step 1 — Calculate Daily Sales Velocity**
```python
velocity['daily_velocity'] = (
    velocity['total_units_sold'] / total_days
)
```
Average units sold per day across the full order history.

**Step 2 — Calculate Days of Stock Remaining**
```python
velocity['days_remaining'] = (
    velocity['simulated_inventory'] / velocity['daily_velocity']
)
```
Current inventory divided by daily velocity.

**Step 3 — Classify Risk Level**
```python
def assign_risk(days):
    if days < 7:
        return 'CRITICAL'
    elif days < 14:
        return 'AT RISK'
    else:
        return 'SAFE'
```

No machine learning required. No expensive tools. 
Just the right question asked of existing data.

---

## Dataset

**Source:** DataCo Smart Supply Chain Dataset
**Platform:** Kaggle
**Size:** 180,519 orders | 53 columns | 2015–2018
**Link:** [DataCo Smart Supply Chain](https://www.kaggle.com/datasets/shashwatwork/dataco-smart-supply-chain-for-big-data-analysis)

To use this project:
1. Download `DataCoSupplyChainDataset.csv` from the link above
2. Place it in the `/data` folder
3. Run the notebook or script

---

## Project Structure

```
inventorypulse/
│
├── data/                          
│   └── README.md                  
├── notebooks/
│   └── stockout_risk_analysis.ipynb   
├── outputs/
│   ├── stockout_risk_dashboard.png    
│   ├── risk_table_full.csv            
│   └── risk_table_critical.csv        
├── src/
│   └── risk_model.py              
├── assets/
│   └── dashboard_preview.png      
├── README.md                      
└── requirements.txt               
```

---

## Tech Stack

| Tool | Purpose |
|------|---------|
| Python 3.x | Core analysis |
| pandas | Data manipulation |
| numpy | Numerical calculations |
| matplotlib | Dashboard visualization |

---

## Installation

```bash
# Clone the repo
git clone https://github.com/kamranrazzaq185-netizen/inventorypulse.git

# Navigate to project
cd inventorypulse

# Install dependencies
pip install -r requirements.txt

# Run the analysis
python src/risk_model.py
```

---

## Key Insights

**1. Stockouts are predictable.**
Sales velocity modeling provides up to 14 days of 
advance warning before a product runs out of stock.

**2. The data already exists.**
No new data collection is required. Every order 
management system contains the inputs for this model.

**3. 1 in 5 SKUs needed attention.**
17.8% of analyzed SKUs were either critical or at risk 
at the time of analysis — with zero active alerts.

**4. High velocity products are the biggest risk.**
Fast-moving SKUs deplete inventory faster than 
replenishment cycles can compensate without early warning.

---

## Next Steps / Roadmap

- [ ] Connect to live inventory feed via API
- [ ] Add automated email alerts for critical SKUs
- [ ] Build Power BI version of dashboard
- [ ] Add reorder point calculation
- [ ] Extend to multi-warehouse analysis

---

## About

Built by **Kamran Razzaq**
Supply Chain Analyst | Python | Power BI | SQL | Excel

Focused on building AI-assisted supply chain analytics 
systems that turn operational data into forward-looking decisions.

[LinkedIn](https://linkedin.com/in/kamranrazzaq) | 
Open to remote Supply Chain Analyst opportunities

---

## License

MIT License — free to use and adapt with attribution.
