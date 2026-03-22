"""
Neural Translation Visualizer - Modern Professional Design
Clean, minimal, presentation-ready interface
"""

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import time
import textwrap
import html
from model_wrapper import TranslationVisualizer

# Page configuration
st.set_page_config(
    page_title="Neural Translation Visualizer",
    page_icon="🌐",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Modern, clean CSS
st.markdown("""
<style>
    /* Clean base */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* Clean white background */
    .stApp {
        background: #f8f9fa;
    }

    /* Modern typography */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    * {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    }

    /* Header */
    .modern-header {
        text-align: center;
        padding: 2rem 0 1rem 0;
        margin-bottom: 2rem;
    }

    .modern-header h1 {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1a1a1a;
        margin-bottom: 0.5rem;
        letter-spacing: -0.02em;
    }

    .modern-header p {
        font-size: 1.1rem;
        color: #6b7280;
        font-weight: 400;
    }

    /* Clean card design */
    .visual-card {
        background: white;
        border-radius: 16px;
        padding: 2rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.06), 0 1px 2px rgba(0,0,0,0.04);
        margin-bottom: 1.5rem;
        border: 1px solid #e5e7eb;
        min-height: 450px;
    }

    /* Card for plotly charts */
    .stPlotlyChart {
        background: white;
        border-radius: 16px;
        padding: 1.5rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.06), 0 1px 2px rgba(0,0,0,0.04);
        border: 1px solid #e5e7eb;
        margin-top: 1rem;
    }

    .info-card {
        background: white;
        border-radius: 16px;
        padding: 1.75rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.06), 0 1px 2px rgba(0,0,0,0.04);
        margin-bottom: 1.5rem;
        border: 1px solid #e5e7eb;
        border-left: 4px solid #3b82f6;
    }

    /* Stage badge */
    .stage-badge {
        display: inline-block;
        background: #3b82f6;
        color: white;
        padding: 0.5rem 1.25rem;
        border-radius: 9999px;
        font-weight: 600;
        font-size: 0.875rem;
        margin-bottom: 1.5rem;
        letter-spacing: 0.025em;
        text-transform: uppercase;
    }

    /* Modern tokens */
    .token-modern {
        display: inline-block;
        padding: 0.5rem 1rem;
        margin: 0.375rem;
        border-radius: 8px;
        font-weight: 500;
        font-size: 0.95rem;
        transition: all 0.2s ease;
        border: 1.5px solid;
    }

    .token-input {
        background: #eff6ff;
        border-color: #3b82f6;
        color: #1e40af;
    }

    .token-output {
        background: #fef3c7;
        border-color: #f59e0b;
        color: #92400e;
    }

    .token-modern:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }

    /* Section titles */
    .section-title {
        font-size: 1.25rem;
        font-weight: 600;
        color: #1a1a1a;
        margin-bottom: 1.25rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }

    /* Info box */
    .info-title {
        font-size: 1.125rem;
        font-weight: 600;
        color: #1a1a1a;
        margin-bottom: 1rem;
    }

    .info-text {
        font-size: 0.95rem;
        line-height: 1.7;
        color: #4b5563;
    }

    /* Progress dots */
    .progress-container {
        text-align: center;
        padding: 1.5rem 0;
        margin: 1rem 0;
    }

    .progress-dot {
        height: 10px;
        width: 10px;
        margin: 0 6px;
        background-color: #d1d5db;
        border-radius: 50%;
        display: inline-block;
        transition: all 0.3s ease;
    }

    .progress-dot.active {
        background-color: #3b82f6;
        width: 32px;
        border-radius: 5px;
    }

    /* Stats */
    .stat-row {
        display: flex;
        justify-content: space-around;
        margin: 1.5rem 0;
        padding: 1.25rem;
        background: #f9fafb;
        border-radius: 12px;
    }

    .stat-item {
        text-align: center;
    }

    .stat-value {
        font-size: 1.875rem;
        font-weight: 700;
        color: #1a1a1a;
        display: block;
    }

    .stat-label {
        font-size: 0.875rem;
        color: #6b7280;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-top: 0.25rem;
    }

    /* Architecture boxes */
    .arch-layer {
        background: white;
        border: 2px solid #e5e7eb;
        border-radius: 12px;
        padding: 1.25rem;
        margin: 0.75rem 0;
        text-align: center;
        font-weight: 600;
        color: #4b5563;
        transition: all 0.3s ease;
    }

    .arch-layer.active {
        background: #3b82f6;
        color: white;
        border-color: #3b82f6;
        transform: translateX(8px);
        box-shadow: 0 4px 6px rgba(59, 130, 246, 0.3);
    }

    /* Attention bars */
    .attention-item {
        margin: 0.75rem 0;
        padding: 0.5rem 0;
    }

    .attention-label {
        display: inline-block;
        width: 120px;
        font-weight: 500;
        color: #4b5563;
        font-size: 0.9rem;
    }

    .attention-bar-bg {
        display: inline-block;
        width: calc(100% - 180px);
        height: 28px;
        background: #f3f4f6;
        border-radius: 6px;
        overflow: hidden;
        vertical-align: middle;
    }

    .attention-bar-fill {
        height: 100%;
        background: linear-gradient(90deg, #3b82f6, #60a5fa);
        border-radius: 6px;
        transition: width 0.5s ease;
        display: flex;
        align-items: center;
        justify-content: flex-end;
        padding-right: 8px;
    }

    .attention-score {
        display: inline-block;
        width: 50px;
        text-align: right;
        font-weight: 600;
        color: #6b7280;
        font-size: 0.85rem;
        vertical-align: middle;
        margin-left: 8px;
    }

    /* Generation display */
    .gen-current {
        background: #f9fafb;
        border: 2px dashed #d1d5db;
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
        font-size: 1.25rem;
        color: #4b5563;
        min-height: 80px;
        display: flex;
        align-items: center;
        justify-content: center;
    }

    .gen-next {
        background: linear-gradient(135deg, #3b82f6, #2563eb);
        color: white;
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
        font-size: 1.5rem;
        font-weight: 600;
        text-align: center;
        box-shadow: 0 4px 6px rgba(59, 130, 246, 0.25);
    }

    /* Arrow */
    .flow-arrow {
        text-align: center;
        font-size: 2rem;
        color: #9ca3af;
        margin: 1rem 0;
    }

    /* Final result */
    .translation-box {
        background: linear-gradient(135deg, #10b981, #059669);
        color: white;
        border-radius: 12px;
        padding: 2rem;
        margin: 1.5rem 0;
        font-size: 1.5rem;
        font-weight: 500;
        text-align: center;
        box-shadow: 0 4px 6px rgba(16, 185, 129, 0.25);
    }

    /* Welcome screen */
    .welcome-card {
        background: white;
        border-radius: 16px;
        padding: 4rem 3rem;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        margin: 4rem auto;
        max-width: 800px;
        border: 1px solid #e5e7eb;
    }

    .welcome-card h2 {
        font-size: 2rem;
        font-weight: 700;
        color: #1a1a1a;
        margin-bottom: 1rem;
    }

    .welcome-card p {
        font-size: 1.1rem;
        color: #6b7280;
        line-height: 1.7;
    }

    /* Buttons */
    .stButton>button {
        border-radius: 8px;
        font-weight: 600;
        padding: 0.5rem 1.5rem;
        transition: all 0.2s ease;
    }

    /* Hide streamlit branding */
    .css-1v0mbdj {
        display: none;
    }

    /* Formula */
    .formula-display {
        background: #f9fafb;
        border: 1px solid #e5e7eb;
        border-radius: 8px;
        padding: 1.5rem;
        margin: 1rem 0;
        text-align: center;
        font-family: 'Courier New', monospace;
    }

    /* Highlight */
    .highlight {
        background: #fef3c7;
        padding: 2px 6px;
        border-radius: 4px;
        font-weight: 600;
        color: #92400e;
    }

    /* Clean list */
    .info-list {
        list-style: none;
        padding: 0;
    }

    .info-list li {
        padding: 0.5rem 0;
        padding-left: 1.5rem;
        position: relative;
        color: #4b5563;
    }

    .info-list li:before {
        content: "→";
        position: absolute;
        left: 0;
        color: #3b82f6;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Session state
if 'visualizer' not in st.session_state:
    st.session_state.visualizer = None
if 'translation_result' not in st.session_state:
    st.session_state.translation_result = None
if 'current_stage' not in st.session_state:
    st.session_state.current_stage = 0
if 'animation_idx' not in st.session_state:
    st.session_state.animation_idx = 0

# @st.cache_resource
def load_model():
    # Force reload of the class definition if it changed
    import model_wrapper
    import importlib
    importlib.reload(model_wrapper)
    from model_wrapper import TranslationVisualizer
    
    viz = TranslationVisualizer()
    viz.load_model()
    return viz

# Examples
EXAMPLES = {
    "Government": ">>tam<< The government has announced new rules for public safety.",
    "Simple": ">>tam<< The cat sits on the mat.",
    "Technology": ">>tam<< Technology is changing our world.",
    "Question": ">>tam<< How are you today?",
}

STAGES = [
    "Input & Tokenization",
    "Embedding Vectors",
    "Encoder Processing",
    "Attention Mechanism",
    "Output Generation",
    "Final Translation"
]

def render_progress_dots(current):
    html = '<div class="progress-container">'
    for i in range(len(STAGES)):
        active = "active" if i == current else ""
        html += f'<span class="progress-dot {active}"></span>'
    html += '</div>'
    st.markdown(html, unsafe_allow_html=True)

def render_stage_0(result):
    """Tokenization"""
    st.markdown('<span class="stage-badge">📝 Stage 1: Input & Tokenization</span>', unsafe_allow_html=True)

    col1, col2 = st.columns([6, 4])

    with col1:
        # Use native Streamlit container
        with st.container():
            st.markdown("#### 🔤 Breaking Text into Tokens")

            # Display tokens using columns
            tokens_to_show = result['input_tokens'][:st.session_state.animation_idx + 1]

            # Group tokens into rows of 5
            for i in range(0, len(tokens_to_show), 5):
                cols = st.columns(5)
                for j, col in enumerate(cols):
                    if i + j < len(tokens_to_show):
                        token = tokens_to_show[i + j]
                        clean = token.replace('▁', '').strip() or '·'
                        with col:
                            st.markdown(f'<div style="background: #eff6ff; border: 1.5px solid #3b82f6; border-radius: 8px; padding: 0.5rem; text-align: center; color: #1e40af; font-weight: 500; margin: 0.25rem 0;">{html.escape(clean)}</div>', unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            # Stats row
            col_a, col_b = st.columns(2)
            with col_a:
                st.metric("Total Tokens", len(result['input_tokens']))
            with col_b:
                st.metric("Dimensions", result['architecture_info']['hidden_size'])

        if st.session_state.animation_idx < len(result['input_tokens']) - 1:
            st.session_state.animation_idx += 1
            time.sleep(0.25)
            st.rerun()

    with col2:
        # Use native Streamlit components instead of HTML
        st.markdown("### 💡 What are Tokens?")

        st.markdown("""
        **Tokens are the basic units of language that AI models understand.**
        Instead of processing entire words or individual characters, the model breaks text into optimal chunks called tokens.
        """)

        st.markdown("**Why not just use words?**")
        st.markdown(f"""
        - **Flexibility**: Handles unknown words by breaking them into known parts
        - **Efficiency**: Common words = 1 token, rare words = multiple tokens
        - **Language coverage**: Works across different languages and scripts
        - **Vocabulary size**: Keeps the model's vocabulary manageable (~60,000 tokens vs millions of possible words)
        """)

        st.markdown("**Special tokens:**")
        st.markdown("""
        - **>>tam<<**: Tells the model to translate to Tamil
        - **[space]**: Underscore (_) represents word boundaries
        - **<pad>**: Padding tokens for batch processing
        """)

        st.info('**Example:** "government" might be one token, but "unprecedented" could be split into "un" + "precedent" + "ed"')

def render_stage_1(result):
    """Embedding"""
    st.markdown('<span class="stage-badge">🧮 Stage 2: Embedding Vectors</span>', unsafe_allow_html=True)

    col1, col2 = st.columns([6, 4])

    with col1:
        sample_idx = min(2, len(result['input_tokens']) - 1)
        sample_token = result['input_tokens'][sample_idx].replace('▁', '').strip()

        # Use native Streamlit container
        with st.container():
            st.markdown("#### 🎯 Token → Vector Conversion")

            st.markdown(f"""
            Converting the token "**{sample_token}**" into a **{result['architecture_info']['hidden_size']}-dimensional** vector.

            Each bar below represents one dimension of the embedding vector.
            These numbers capture the semantic meaning of the word.
            """)

            # Create the embedding visualization
            try:
                embeddings = st.session_state.visualizer.get_token_embeddings([result['input_tokens'][sample_idx]])
                embedding_vector = embeddings[0][:24]

                fig = go.Figure(data=[
                    go.Bar(
                        x=list(range(24)),
                        y=embedding_vector.tolist(),
                        marker=dict(
                            color='#3b82f6',
                            line=dict(color='#2563eb', width=1)
                        ),
                        hovertemplate='Dimension %{x}<br>Value: %{y:.3f}<extra></extra>'
                    )
                ])
                fig.update_layout(
                    title=f"First 24 of {result['architecture_info']['hidden_size']} dimensions",
                    height=350,
                    margin=dict(l=40, r=20, t=50, b=40),
                    plot_bgcolor='white',
                    paper_bgcolor='white',
                    xaxis=dict(title="Dimension Index", showgrid=False),
                    yaxis=dict(title="Embedding Value", showgrid=True, gridcolor='#f3f4f6'),
                    font=dict(family="Inter", size=11)
                )
                st.plotly_chart(fig, use_container_width=True, key=f"embed_{st.session_state.current_stage}")

            except Exception as e:
                st.info(f'The token "**{sample_token}**" is represented as a vector of {result["architecture_info"]["hidden_size"]} floating-point numbers.')

    with col2:
        st.markdown("### 💡 Understanding Embeddings")

        st.markdown(f"""
        **What is an embedding?**

        It's a way to represent words as vectors (lists of numbers).
        Each token gets converted into exactly **{result['architecture_info']['hidden_size']} numbers**.
        """)

        st.markdown("**Why vectors?**")
        st.markdown("""
        - **Semantic similarity**: Words with similar meanings have similar vectors
        - **Math operations**: We can do calculations on meaning itself
        - **Learned patterns**: These numbers are learned during training on millions of sentences
        - **Context-free**: This is just the starting point - later layers add context
        """)

        st.success('**Example:** The vector for "king" - "man" + "woman" ≈ "queen"')

def render_stage_2(result):
    """Encoder"""
    st.markdown('<span class="stage-badge">⚡ Stage 3: Encoder Processing</span>', unsafe_allow_html=True)

    col1, col2 = st.columns([6, 4])

    with col1:
        with st.container():
            st.markdown("#### 🏗️ Encoder Layers")

            # Display encoder layers
            for i in range(result['architecture_info']['encoder_layers']):
                if i <= st.session_state.animation_idx:
                    # Active layer with blue background
                    st.markdown(textwrap.dedent(f"""
                    <div style="background: #3b82f6; color: white; border: 2px solid #3b82f6;
                    border-radius: 12px; padding: 1.25rem; margin: 0.75rem 0; text-align: center;
                    font-weight: 600; box-shadow: 0 4px 6px rgba(59, 130, 246, 0.3);">
                        Layer {i+1} · Self-Attention + Feed-Forward
                    </div>
                    """), unsafe_allow_html=True)
                else:
                    # Inactive layer with gray border
                    st.markdown(textwrap.dedent(f"""
                    <div style="background: white; color: #4b5563; border: 2px solid #e5e7eb;
                    border-radius: 12px; padding: 1.25rem; margin: 0.75rem 0; text-align: center;
                    font-weight: 600;">
                        Layer {i+1} · Self-Attention + Feed-Forward
                    </div>
                    """), unsafe_allow_html=True)

        if st.session_state.animation_idx < result['architecture_info']['encoder_layers'] - 1:
            st.session_state.animation_idx += 1
            time.sleep(0.3)
            st.rerun()

    with col2:
        st.markdown("### 💡 The Encoder's Job")

        st.markdown(f"""
        **The encoder processes the input text** to create rich, contextualized representations.
        It has **{result['architecture_info']['encoder_layers']} layers**, each making the representations more sophisticated.
        """)

        st.markdown("**What happens in each layer?**")
        st.markdown("""
        - **Self-Attention**: Each word "looks" at all other words to understand context

          _Example: In "bank account", "bank" pays attention to "account" to know it's a financial institution, not a river bank_

        - **Feed-Forward Network**: Applies non-linear transformations to extract features

        - **Layer Normalization**: Keeps values stable and prevents numerical issues

        - **Residual Connections**: Allows information to flow directly through layers
        """)

        st.markdown("**Why multiple layers?**")
        st.markdown("""
        - **Layer 1-2**: Learn basic syntax and grammar
        - **Layer 3-4**: Understand semantic relationships
        - **Layer 5-6**: Capture complex contextual meaning
        """)

        st.info("By the end, each token's representation contains information about the entire sentence's meaning and structure.")

def render_stage_3(result):
    """Attention"""
    st.markdown('<span class="stage-badge">🎯 Stage 4: Attention Mechanism</span>', unsafe_allow_html=True)

    col1, col2 = st.columns([6, 4])

    with col1:
        with st.container():
            st.markdown("#### 🔍 Cross-Attention Weights")

            if len(result['attention_weights']) > 0:
                target_idx = min(st.session_state.animation_idx, len(result['output_tokens']) - 1)
                if target_idx < len(result['attention_weights']):
                    attention_scores = result['attention_weights'][target_idx]
                    output_token = result['output_tokens'][target_idx].replace('▁', '').strip()

                    st.markdown(f"**Generating:** {output_token}")
                    st.markdown("")

                    # Display attention bars using Streamlit progress bars
                    for token, score in zip(result['input_tokens'], attention_scores):
                        clean = token.replace('▁', '').strip() or '·'

                        col_label, col_bar, col_score = st.columns([2, 6, 1])
                        with col_label:
                            st.markdown(f"**{clean}**")
                        with col_bar:
                            st.progress(float(score))
                        with col_score:
                            st.markdown(f"`{score:.2f}`")

        if st.session_state.animation_idx < min(4, len(result['output_tokens']) - 1):
            st.session_state.animation_idx += 1
            time.sleep(0.7)
            st.rerun()

    with col2:
        st.markdown("### 💡 Cross-Attention Explained")

        st.markdown("""
        **Attention is the "magic" of transformers.**
        When generating each output word, the decoder "attends" to (focuses on) the most relevant input words.
        """)

        st.markdown("**How does it work?**")
        st.markdown("""
        - **Query (Q)**: "What am I looking for?" - from the decoder
        - **Key (K)**: "What information do I have?" - from the encoder
        - **Value (V)**: "The actual content" - from the encoder
        - **Softmax**: Converts scores to probabilities (bars you see → they sum to 1.0)
        """)

        st.info("""
        **Reading the visualization:**
        - Longer blue bars = stronger attention (focus)
        - The model focuses on different input words for each output word
        - Numbers show exact attention weights (0.0 to 1.0)
        """)

        st.markdown("**The Attention Formula:**")
        st.latex(r"\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) V")

        st.markdown("""
        This mathematical formula computes how much each input word should influence the current output word being generated.
        """)

        st.success('**Fun fact:** This attention mechanism is why the groundbreaking 2017 paper was titled "Attention Is All You Need" - it revolutionized NLP!')

def render_stage_4(result):
    """Generation"""
    st.markdown('<span class="stage-badge">⚡ Stage 5: Output Generation</span>', unsafe_allow_html=True)

    col1, col2 = st.columns([6, 4])

    with col1:
        with st.container():
            st.markdown("#### 🎬 Token-by-Token Generation")

            if st.session_state.animation_idx < len(result['step_by_step']):
                step = result['step_by_step'][st.session_state.animation_idx]

                st.markdown(f"**Step {step['step']} of {len(result['step_by_step'])}**")
                st.markdown("")

                # Current text generated so far
                st.markdown("**Generated so far:**")
                st.markdown(textwrap.dedent(f"""
                <div style="background: #f9fafb; border: 2px dashed #d1d5db; border-radius: 12px;
                padding: 1.5rem; margin: 1rem 0; font-size: 1.25rem; color: #4b5563; min-height: 80px;
                display: flex; align-items: center; justify-content: center;">
                    {html.escape(step["generated_so_far"]) or "[start]"}
                </div>
                """), unsafe_allow_html=True)

                st.markdown("<div style='text-align: center; font-size: 2rem; color: #9ca3af; margin: 1rem 0;'>↓</div>", unsafe_allow_html=True)

                # Next predicted token
                next_token_display = html.escape(step["next_token"])
                st.markdown("**Next token:**")
                st.markdown(textwrap.dedent(f"""
                <div style="background: linear-gradient(135deg, #3b82f6, #2563eb); color: white;
                border-radius: 12px; padding: 1.5rem; margin: 1rem 0; font-size: 1.5rem;
                font-weight: 600; text-align: center; box-shadow: 0 4px 6px rgba(59, 130, 246, 0.25);">
                    {next_token_display}
                </div>
                """), unsafe_allow_html=True)

                st.markdown("**Top predictions:**")
                for token, prob in step['top_predictions'][:3]:
                    st.markdown(f"• **{token}**: {prob:.1%}")

        if st.session_state.animation_idx < len(result['step_by_step']) - 1:
            st.session_state.animation_idx += 1
            time.sleep(0.8)
            st.rerun()

    with col2:
        st.markdown("### 💡 Autoregressive Generation")

        st.markdown("""
        **The model generates translation one token at a time**, in a sequential process called "autoregressive" generation.
        """)

        st.markdown("**Step-by-step process:**")
        st.markdown("""
        1. **Step 1**: Start with the input and a "start" token
        2. **Step 2**: Predict the first output token using attention
        3. **Step 3**: Add that token to the sequence
        4. **Step 4**: Use ALL previous tokens to predict the next one
        5. **Repeat**: Continue until we generate an "end" token
        """)

        st.markdown("**Why 'autoregressive'?**")
        st.markdown("""
        Each new token depends on (regresses on) previously generated tokens.
        The model uses its own outputs as inputs for the next step.
        """)

        st.markdown("**Top predictions shown:**")
        st.markdown("""
        - **1st choice**: Most likely next token (highest probability)
        - **2nd & 3rd**: Alternative options the model considered
        - **Beam search**: The model actually explores multiple paths simultaneously
        """)

        st.warning("**Note:** This is why translation can be slow - each token requires a full forward pass through the decoder!")

def render_stage_5(result):
    """Final"""
    st.markdown('<span class="stage-badge">🎉 Stage 6: Final Translation</span>', unsafe_allow_html=True)

    col1, col2 = st.columns([6, 4])

    with col1:
        # Build tokens HTML
        input_html = ""
        for token in result['input_tokens']:
            clean = token.replace('▁', '').strip() or '·'
            input_html += f'<span class="token-modern token-input">{html.escape(clean)}</span>'

        output_html = ""
        for token in result['output_tokens']:
            clean = token.replace('▁', '').strip() or '·'
            output_html += f'<span class="token-modern token-output">{html.escape(clean)}</span>'

        full_html = textwrap.dedent(f'''
        <div class="visual-card">
            <div class="section-title">✅ Complete Translation</div>

            <p style='font-size: 0.9rem; color: #6b7280; margin: 1.5rem 0 0.5rem 0;'>Input (English):</p>
            <div style="margin-bottom: 1.5rem;">
                {input_html}
            </div>

            <div class="flow-arrow">⬇</div>

            <p style='font-size: 0.9rem; color: #6b7280; margin: 1rem 0 0.5rem 0;'>Output (Tamil):</p>
            <div style="margin-bottom: 1.5rem;">
                {output_html}
            </div>

            <div class="translation-box">{result["translation"]}</div>
        </div>
        ''')
        st.markdown(full_html, unsafe_allow_html=True)

    with col2:
        st.markdown("### 📊 Translation Summary")

        st.markdown("**The complete translation pipeline:**")

        st.markdown(f"""
        **1. Tokenization**
        - {len(result['input_tokens'])} English tokens → {len(result['output_tokens'])} Tamil tokens

        **2. Embedding**
        - Each token → {result['architecture_info']['hidden_size']}-dimensional vector

        **3. Encoder**
        - {result['architecture_info']['encoder_layers']} layers of self-attention processing

        **4. Cross-Attention**
        - Decoder focuses on relevant input words

        **5. Generation**
        - {len(result['step_by_step'])} sequential decoding steps
        """)

        st.markdown("**Model Architecture:**")
        st.markdown(f"""
        - Encoder layers: **{result['architecture_info']['encoder_layers']}**
        - Decoder layers: **{result['architecture_info']['decoder_layers']}**
        - Hidden dimensions: **{result['architecture_info']['hidden_size']}**
        - Vocabulary size: **{result['architecture_info']['vocab_size']:,}** tokens
        - Total parameters: **~74 million**
        """)

        st.info("""
        **Model:** Helsinki-NLP/opus-mt-en-dra

        **Architecture:** MarianMT (Transformer)

        **Training:** Millions of English-Dravidian sentence pairs
        """)

# Header
st.markdown('''
<div class="modern-header">
    <h1>🌐 Neural Translation Visualizer</h1>
    <p>Watch AI translate English to Tamil in real-time</p>
</div>
''', unsafe_allow_html=True)

# Controls
col1, col2, col3, col4 = st.columns([3, 3, 1, 1])
with col1:
    selected = st.selectbox("Select Example", list(EXAMPLES.keys()), label_visibility="collapsed")
with col2:
    text_input = st.text_input("Or enter custom", value=EXAMPLES[selected], label_visibility="collapsed")
with col3:
    if st.button("Start", use_container_width=True):
        st.session_state.translation_result = None # Clear previous
        with st.spinner("Processing..."):
            try:
                viz = load_model()
                st.session_state.visualizer = viz
                
                # Perform translation
                result = viz.translate_with_details(text_input)
                st.session_state.translation_result = result
                
                st.session_state.current_stage = 0
                st.session_state.animation_idx = 0
                st.rerun()
            except Exception as e:
                st.error(f"Error: {e}")
with col4:
    if st.button("Reset", use_container_width=True):
        st.session_state.current_stage = 0
        st.session_state.animation_idx = 0
        st.session_state.translation_result = None
        st.session_state.visualizer = None # Clear visualizer to force reload
        st.cache_resource.clear() # Clear resource cache
        st.rerun()

# Main content
if st.session_state.translation_result:
    result = st.session_state.translation_result

    render_progress_dots(st.session_state.current_stage)

    # Navigation
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        if st.button("⏮ First", use_container_width=True):
            st.session_state.current_stage = 0
            st.session_state.animation_idx = 0
            st.rerun()
    with col2:
        if st.button("◀ Prev", use_container_width=True) and st.session_state.current_stage > 0:
            st.session_state.current_stage -= 1
            st.session_state.animation_idx = 0
            st.rerun()
    with col3:
        if st.button("▶ Next", use_container_width=True) and st.session_state.current_stage < len(STAGES) - 1:
            st.session_state.current_stage += 1
            st.session_state.animation_idx = 0
            st.rerun()
    with col4:
        if st.button("⏭ Last", use_container_width=True):
            st.session_state.current_stage = len(STAGES) - 1
            st.session_state.animation_idx = 0
            st.rerun()
    with col5:
        auto = st.checkbox("Auto", key="auto")

    # Render stage
    if st.session_state.current_stage == 0:
        render_stage_0(result)
    elif st.session_state.current_stage == 1:
        render_stage_1(result)
    elif st.session_state.current_stage == 2:
        render_stage_2(result)
    elif st.session_state.current_stage == 3:
        render_stage_3(result)
    elif st.session_state.current_stage == 4:
        render_stage_4(result)
    elif st.session_state.current_stage == 5:
        render_stage_5(result)

    # Auto advance
    if auto and st.session_state.current_stage < len(STAGES) - 1:
        max_idx = len(result['input_tokens'])
        if st.session_state.current_stage == 2:
            max_idx = result['architecture_info']['encoder_layers']
        elif st.session_state.current_stage == 4:
            max_idx = len(result['step_by_step'])

        if st.session_state.animation_idx >= max_idx - 1:
            time.sleep(1.5)
            st.session_state.current_stage += 1
            st.session_state.animation_idx = 0
            st.rerun()

else:
    st.markdown('''
    <div class="welcome-card">
        <h2>👋 Welcome!</h2>
        <p>Select an example or enter your own text, then click <strong>Start</strong> to see how neural translation works.</p>
    </div>
    ''', unsafe_allow_html=True)
