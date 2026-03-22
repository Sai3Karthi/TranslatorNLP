"""
Model Wrapper for Translation Visualization
Handles model loading, translation, and extraction of internal states
"""

import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from typing import Dict, List, Tuple, Any


class TranslationVisualizer:
    """Wraps the translation model and extracts visualization data"""

    def __init__(self, model_name: str = "Helsinki-NLP/opus-mt-en-dra"):
        """
        Initialize the translation model and tokenizer

        Args:
            model_name: HuggingFace model identifier
        """
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def load_model(self):
        """Load the tokenizer and model (cached for performance)"""
        if self.tokenizer is None or self.model is None:
            print(f"Loading model: {self.model_name}...")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(
                self.model_name,
                output_attentions=True,  # Enable attention extraction
                output_hidden_states=True  # Enable hidden state extraction
            )
            self.model.to(self.device)
            self.model.eval()  # Set to evaluation mode
            print("Model loaded successfully!")

        return self.tokenizer, self.model

    def translate_with_details(self, text: str) -> Dict[str, Any]:
        """
        Translate text and extract all visualization data

        Args:
            text: Input English text to translate

        Returns:
            Dictionary containing:
                - translation: Translated text
                - input_tokens: List of input tokens
                - input_ids: List of input token IDs
                - output_tokens: List of output tokens
                - output_ids: List of output token IDs
                - attention_weights: Cross-attention weights (input x output)
                - encoder_attention: Encoder self-attention weights
                - decoder_attention: Decoder self-attention weights
                - step_by_step: Step-by-step generation details
        """
        self.load_model()

        # Tokenize input
        inputs = self.tokenizer(text, return_tensors="pt", padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Get input tokens for visualization
        input_ids = inputs['input_ids'][0].cpu().tolist()
        input_tokens = [self.tokenizer.decode([tid]) for tid in input_ids]

        # Generate translation with attention weights
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=128,
                num_beams=5,  # Beam search for better quality
                output_attentions=True,
                output_scores=True,
                return_dict_in_generate=True
            )

        # Decode translation
        generated_ids = outputs.sequences[0].cpu().tolist()
        translation = self.tokenizer.decode(generated_ids, skip_special_tokens=True)

        # Get output tokens
        output_tokens = [self.tokenizer.decode([tid]) for tid in generated_ids]

        # Extract attention weights (simplified - use last layer cross-attention)
        # For visualization, we'll compute a simpler version by running the model again
        attention_weights = self._extract_attention_weights(inputs, generated_ids)

        # Get step-by-step generation for animation
        step_by_step = self._get_step_by_step_generation(text, max_steps=len(generated_ids))

        # Get model architecture info
        encoder_layers = self.model.config.encoder_layers
        decoder_layers = self.model.config.decoder_layers
        d_model = self.model.config.d_model

        return {
            'translation': translation,
            'input_tokens': input_tokens,
            'input_ids': input_ids,
            'output_tokens': output_tokens,
            'output_ids': generated_ids,
            'attention_weights': attention_weights,
            'step_by_step': step_by_step,
            'architecture_info': {
                'encoder_layers': encoder_layers,
                'decoder_layers': decoder_layers,
                'hidden_size': d_model,
                'vocab_size': self.model.config.vocab_size
            }
        }

    def _extract_attention_weights(self, inputs: Dict, output_ids: List[int]) -> np.ndarray:
        """
        Extract cross-attention weights between input and output tokens

        Returns:
            numpy array of shape (output_length, input_length) with attention scores
        """
        # Run model in teacher-forcing mode to get attention
        try:
            decoder_input_ids = torch.tensor([output_ids]).to(self.device)

            with torch.no_grad():
                outputs = self.model(
                    input_ids=inputs['input_ids'],
                    decoder_input_ids=decoder_input_ids,
                    output_attentions=True
                )

            # Get cross-attention from last decoder layer
            # cross_attentions is a tuple of (num_layers,) each of shape (batch, heads, target_len, source_len)
            if outputs.cross_attentions:
                # Take last layer, average across attention heads
                last_layer_attention = outputs.cross_attentions[-1]  # (1, heads, target_len, source_len)
                attention = last_layer_attention[0].mean(dim=0).cpu().numpy()  # (target_len, source_len)
                return attention
            else:
                # Fallback: create uniform attention
                return np.ones((len(output_ids), inputs['input_ids'].shape[1])) / inputs['input_ids'].shape[1]

        except Exception as e:
            print(f"Warning: Could not extract attention weights: {e}")
            # Return uniform attention as fallback
            return np.ones((len(output_ids), inputs['input_ids'].shape[1])) / inputs['input_ids'].shape[1]

    def _get_step_by_step_generation(self, text: str, max_steps: int = 20) -> List[Dict]:
        """
        Generate translation step by step to show the decoding process

        Returns:
            List of dictionaries, each containing:
                - step: step number
                - generated_so_far: tokens generated up to this step
                - next_token: the token predicted at this step
                - next_token_probs: top-k token probabilities
        """
        self.load_model()

        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)

        # Start with the decoder start token
        decoder_input_ids = torch.tensor([[self.model.config.decoder_start_token_id]]).to(self.device)

        steps = []

        with torch.no_grad():
            for step in range(min(max_steps, 30)):  # Limit steps for performance
                # Get model predictions
                outputs = self.model(
                    input_ids=inputs['input_ids'],
                    decoder_input_ids=decoder_input_ids,
                    output_attentions=False
                )

                # Get logits for next token
                next_token_logits = outputs.logits[0, -1, :]  # (vocab_size,)

                # Mask pad token to prevent generation of pad (common issue with MarianMT)
                if self.model.config.pad_token_id is not None:
                    next_token_logits[self.model.config.pad_token_id] = -float('inf')

                # Also mask the decoder_start_token_id just in case it's different but still problematic
                if self.model.config.decoder_start_token_id is not None:
                     next_token_logits[self.model.config.decoder_start_token_id] = -float('inf')

                # Get probabilities
                probs = torch.softmax(next_token_logits, dim=-1)

                # Get top-5 predictions
                top_probs, top_indices = torch.topk(probs, k=5)
                top_tokens = [self.tokenizer.decode([idx.item()]) for idx in top_indices]
                top_probs_list = top_probs.cpu().tolist()

                # Get the most likely next token
                next_token_id = top_indices[0].item()

                # Store step information
                generated_text = self.tokenizer.decode(decoder_input_ids[0], skip_special_tokens=True)
                steps.append({
                    'step': step + 1,
                    'generated_so_far': generated_text,
                    'next_token': top_tokens[0],
                    'next_token_id': next_token_id,
                    'top_predictions': list(zip(top_tokens, top_probs_list))
                })

                # Check for end token
                if next_token_id == self.tokenizer.eos_token_id:
                    break

                # Append next token for next iteration
                decoder_input_ids = torch.cat([
                    decoder_input_ids,
                    torch.tensor([[next_token_id]]).to(self.device)
                ], dim=1)

        return steps

    def get_model_info(self) -> Dict[str, Any]:
        """Get basic model architecture information"""
        self.load_model()

        config = self.model.config

        return {
            'model_name': self.model_name,
            'encoder_layers': config.encoder_layers,
            'decoder_layers': config.decoder_layers,
            'hidden_size': config.d_model,
            'vocab_size': config.vocab_size,
            'num_attention_heads': config.encoder_attention_heads,
            'feed_forward_dim': config.encoder_ffn_dim,
        }

    def get_token_embeddings(self, tokens: List[str]) -> np.ndarray:
        """
        Get embedding vectors for given tokens

        Args:
            tokens: List of token strings

        Returns:
            numpy array of shape (num_tokens, embedding_dim)
        """
        self.load_model()

        # Get token IDs
        token_ids = self.tokenizer.convert_tokens_to_ids(tokens)

        # Get embeddings from the model
        with torch.no_grad():
            embeddings = self.model.get_encoder().embed_tokens(
                torch.tensor([token_ids]).to(self.device)
            )

        return embeddings[0].cpu().numpy()


# Example usage
if __name__ == "__main__":
    visualizer = TranslationVisualizer()

    # Test translation
    test_text = ">>tam<< The government has announced new rules for public safety."
    result = visualizer.translate_with_details(test_text)

    print(f"Input: {test_text}")
    print(f"Translation: {result['translation']}")
    print(f"Input tokens: {result['input_tokens']}")
    print(f"Output tokens: {result['output_tokens']}")
    print(f"Attention shape: {result['attention_weights'].shape}")
    print(f"Steps: {len(result['step_by_step'])}")
