"""Test script to verify stage data access with sanitization."""
import sys
sys.path.insert(0, '.')

from app.pipeline.storage import storage

pipeline_id = "36141010-bdef-493f-a1b7-d3e15b3246bc"
stage_name_with_slash = "Stemming/Lemmatização #7"

print(f"Testing stage data access...")
print(f"Pipeline ID: {pipeline_id}")
print(f"Stage name: '{stage_name_with_slash}'")
print()

# Try to get data
data = storage.get_stage_data(pipeline_id, stage_name_with_slash)

if data is not None:
    print("✅ SUCCESS! Stage data retrieved")
    print(f"Shape: {data.shape}")
    print(f"Columns: {list(data.columns)}")
    
    # Check for lemmatized column
    lemma_cols = [col for col in data.columns if 'lemmatized' in col.lower()]
    if lemma_cols:
        print(f"\n✅ Lemmatized columns found: {lemma_cols}")
        print(f"Sample data from {lemma_cols[0]}:")
        print(data[lemma_cols[0]].head(3).tolist())
    else:
        print("\n❌ No lemmatized columns found")
else:
    print("❌ FAILED! Stage data not found")
    print("\nAvailable stages:")
    stages = storage.list_stages(pipeline_id)
    for s in stages:
        print(f"  - {s}")
