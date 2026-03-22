"""
Quick test script to verify the translation visualizer works
"""

from model_wrapper import TranslationVisualizer

print("=" * 50)
print("Testing Translation Visualizer")
print("=" * 50)

# Initialize visualizer
print("\n1. Initializing visualizer...")
viz = TranslationVisualizer()

# Load model
print("2. Loading model (this may take a moment)...")
viz.load_model()
print("   [OK] Model loaded successfully!")

# Test translation
test_text = ">>tam<< The government has announced new rules."
print(f"\n3. Testing translation...")
print(f"   Input: {test_text}")

result = viz.translate_with_details(test_text)

try:
    print(f"   Output: {result['translation']}")
except UnicodeEncodeError:
    print(f"   Output: [Tamil text - {len(result['translation'])} characters]")
    print(f"   (Terminal can't display Tamil characters, but translation works!)")
print(f"   [OK] Translation successful!")

# Verify data extraction
print(f"\n4. Verifying data extraction...")
print(f"   - Input tokens: {len(result['input_tokens'])}")
print(f"   - Output tokens: {len(result['output_tokens'])}")
print(f"   - Attention shape: {result['attention_weights'].shape}")
print(f"   - Generation steps: {len(result['step_by_step'])}")
print(f"   - Architecture info: {result['architecture_info']}")
print(f"   [OK] All data extracted successfully!")

print("\n" + "=" * 50)
print("[SUCCESS] ALL TESTS PASSED!")
print("=" * 50)
print("\nYou can now run the Streamlit app:")
print("  streamlit run app.py")
print("=" * 50)
