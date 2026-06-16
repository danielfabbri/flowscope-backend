# 🔄 Pipeline Flow Visualization

## Regression Pipeline: Property Price Prediction

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        PROPERTY PRICE PREDICTION                        │
│                   Identify Undervalued Real Estate                      │
└─────────────────────────────────────────────────────────────────────────┘

📥 STEP 1: Data Ingestion
│  Input: property_sales.json (100 properties, 22 columns)
│  Output: 100 rows × 22 columns
│  ↓ property_id, property_type, bedrooms, sale_price, etc.
│
│
🎯 STEP 2: Column Selection
│  Action: Keep only relevant features
│  Output: 100 rows × 18 columns
│  ↓ Removed: heating_type, parking_type, year_built, days_on_market
│
│
🏷️  STEP 3: Label Encoding (property_type)
│  Action: Apartment=0, Condo=1, House=2, Townhouse=3
│  Output: 100 rows × 18 columns
│  ↓ property_type: "House" → 2
│
│
🏘️  STEP 4: Label Encoding (neighborhood)
│  Action: Downtown=0, Suburbs=1, Uptown=2, etc.
│  Output: 100 rows × 18 columns
│  ↓ neighborhood: "Riverside" → 4
│
│
⭐ STEP 5: Label Encoding (condition)
│  Action: Poor=0, Fair=1, Good=2, Excellent=3
│  Output: 100 rows × 18 columns
│  ↓ condition: "Excellent" → 3
│
│
📏 STEP 6: Normalization (StandardScaler)
│  Action: Normalize numeric features (mean=0, std=1)
│  Output: 100 rows × 18 columns
│  ↓ square_feet: 2500 → 0.543, 1200 → -0.892
│
│
🤖 STEP 7: ML Modeling (Random Forest Regression)
│  Action: Train model to predict sale_price
│  Split: 80 train / 20 test
│  Output: 100 rows × 22 columns (added 4 new columns)
│  ↓ New columns:
│    • predicted_price (model prediction)
│    • price_difference (actual - predicted)
│    • model_mae, model_rmse, model_r2_score (metrics)
│    • importance_square_feet, importance_neighborhood, etc. (top 10 features)
│  ↓ Metrics: R²=0.88, MAE=$28,000
│
│
💰 STEP 8: Feature Creation (price_gap)
│  Action: Calculate price_gap = predicted_price - sale_price
│  Output: 100 rows × 23 columns
│  ↓ New column: price_gap
│    • Positive = Undervalued (good deal!)
│    • Negative = Overvalued
│
│
🎯 STEP 9: Row Filtering (price_gap > $20,000)
│  Action: Keep only undervalued properties
│  Output: ~25 rows × 23 columns (75 filtered out)
│  ↓ Only properties with price_gap > $20,000
│
│
📞 STEP 10: Data Enrichment (JOIN with property_details.csv)
│  Action: Add agent contact information
│  Output: ~25 rows × 28 columns (added 5 new columns)
│  ↓ New columns:
│    • agent_name
│    • agent_phone
│    • agent_email
│    • listing_url
│
│
💾 STEP 11: Output (CSV Export)
│  Action: Save results to CSV file
│  Output: data/output/undervalued_properties.csv
│  ↓ Final dataset with ~25 undervalued properties
│
└───────────────────────────────────────────────────────────────────────┘
     ✅ PIPELINE COMPLETE! 🎉
     
     Results: 25 undervalued properties ready for review
     Each property is priced $20K+ below predicted value
     Contact info included for immediate follow-up
```

---

## 📊 Data Transformation Example

### Before Pipeline (Raw Data)
```
property_id: P045
property_type: "House"
neighborhood: "Riverside"
bedrooms: 4
bathrooms: 2.5
square_feet: 2800
condition: "Good"
sale_price: $320,000
```

### After ML Modeling
```
property_id: P045
property_type: 2 (encoded)
neighborhood: 4 (encoded)
bedrooms: 4
bathrooms: 2.5
square_feet: 0.712 (normalized)
condition: 2 (encoded)
sale_price: $320,000
predicted_price: $375,000 ← MODEL PREDICTION
price_gap: $55,000 ← NEW COLUMN
model_r2_score: 0.88
```

### After Enrichment (Final)
```
property_id: P045
property_type: 2
neighborhood: 4
sale_price: $320,000
predicted_price: $375,000
price_gap: $55,000 ← GREAT DEAL!
agent_name: "Sarah Johnson" ← JOINED
agent_phone: "+1-555-2045" ← JOINED
agent_email: "sarah.johnson@realty.com" ← JOINED
listing_url: "https://realty.example.com/listing/p045" ← JOINED
```

---

## 🎯 Key Insights

**Data Size Changes:**
- Start: 100 properties
- After Filtering: ~25 properties (75% filtered out)
- Only the best deals remain!

**Columns Added:**
- Step 7 (ML): +4 columns (predictions + metrics)
- Step 8 (Feature): +1 column (price_gap)
- Step 10 (Enrich): +5 columns (agent info)

**Total Pipeline Time:** ~3-5 seconds
- Data prep: 1s
- ML training: 2s
- Filtering + Enrichment: 1s

---

## 💡 What You'll See Live

As the pipeline runs, you'll see each step complete with:
- ✅ Step name turning green
- 📊 Row count updating
- 🔢 Column count increasing
- ⚡ Real-time progress updates

Click on any completed step to see the data at that stage!
