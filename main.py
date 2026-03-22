# 1. Import the necessary classes from the transformers library
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Define the model name from Hugging Face
model_name = "Helsinki-NLP/opus-mt-en-dra"

# 2. Load the Tokenizer
# The tokenizer prepares the input text for the model.
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_name)
print("Tokenizer loaded successfully.")

# 3. Load the Model
# This is the core translation model.
print("Loading model... (This may take a moment)")
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
print("Model loaded successfully.")

# 4. Prepare the text for translation
english_text = ">>tam<< The government has announced new rules for public safety."

# Use the tokenizer to convert the English text to a format the model understands (input IDs)
# 'return_tensors="pt"' tells the tokenizer to return PyTorch tensors.
inputs = tokenizer(english_text, return_tensors="pt")

# 5. Generate the translation
# The model takes the input IDs and generates the output IDs for the translated text.
generated_ids = model.generate(**inputs)
# The 'generate' function creates the sequence of token IDs for the output.
# The '**' before 'inputs' is a neat Python trick to unpack the dictionary from the tokenizer.

# 6. Decode the translated text
# We use the same tokenizer to convert the output IDs back into human-readable text.
tamil_translation = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
# 'skip_special_tokens=True' removes any special model tokens (like <pad> or </s>) from the output.
# We take the first element [0] because the function can work on batches of text.

# 7. Print the result
print("\n=============================================")
print(f"Original English Text: {english_text}")
print(f"Translated Tamil Text: {tamil_translation}")
print("=============================================")