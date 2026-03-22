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
    /* PROJECTOR MODE - MASSIVE FONTS */
    html {
        font-size: 24px; /* Base font size boost */
    }

    /* Hide standard Streamlit UI */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Global Layout - No Scroll */
    .stApp {
        background: #f8fafc; /* Lighter background */
        /* Allow scroll if needed but hide scrollbar to prevent "stuck" UI if content overflows */
        height: 100vh;
        overflow-y: auto; 
    }
    
    /* Hide scrollbar for Chrome, Safari and Opera */
    .stApp::-webkit-scrollbar {
        display: none;
    }

    /* Hide scrollbar for IE, Edge and Firefox */
    .stApp {
        -ms-overflow-style: none;  /* IE and Edge */
        scrollbar-width: none;  /* Firefox */
    }

    .block-container {
        max-width: 100vw !important;
        padding-top: 0.5rem !important; 
        padding-bottom: 0rem !important;
        padding-left: 2rem !important;
        padding-right: 2rem !important;
        /* Ensure we use full height but don't force overflow hidden globally to avoid cutting off nav */
        min-height: 100vh;
        display: flex;
        flex-direction: column;
    }

    /* Typography Overrides for Projector */
    h1 { font-size: 3rem !important; margin-bottom: 0.5rem !important; margin-top: 0 !important; }
    h2 { font-size: 2.5rem !important; margin-bottom: 0.5rem !important; }
    h3 { font-size: 2rem !important; margin-bottom: 0.5rem !important; }
    h4 { font-size: 1.5rem !important; margin-bottom: 0.5rem !important; }
    p, li, span, div, label { font-size: 1.3rem; line-height: 1.5; }
    
    /* Specific overrides for readable text */
    .markdown-text-container { font-size: 1.3rem !important; }
    
    /* FIX: Input Fields Sizing for Large Fonts */
    .stTextInput input {
        font-size: 1.4rem !important;
        height: 3.5rem !important;
        padding: 0.5rem 1rem !important;
    }
    
    .stSelectbox div[data-baseweb="select"] {
        height: 3.5rem !important;
    }
    
    .stSelectbox div[data-baseweb="select"] > div {
        font-size: 1.4rem !important;
        line-height: 3.5rem !important;
    }

    /* Fix dropdown menu items */
    ul[data-testid="stSelectboxVirtualDropdown"] li {
        font-size: 1.3rem !important;
        height: auto !important;
        min-height: 3rem !important;
        display: flex;
        align-items: center;
        padding-top: 0.5rem !important;
        padding-bottom: 0.5rem !important;
    }
    
    /* Code/Token styling */
    .token-modern {
        font-size: 1.6rem !important;
        padding: 0.4rem 0.8rem !important;
        margin: 0.2rem !important;
        display: inline-block;
        border-radius: 8px;
    }

    /* Container Styling - The "Slide" Card */
    div[data-testid="stVerticalBlockBorderWrapper"] {
        background: white;
        border-radius: 24px;
        padding: 2rem; /* Reduced padding */
        box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.1);
        border: 2px solid #e2e8f0;
        height: 78vh !important; /* Adjusted height */
        overflow-y: auto; 
        display: flex;
        flex-direction: column;
        margin-bottom: 1vh;
    }
    
    /* Buttons */
    .stButton > button {
        font-size: 1.4rem !important;
        padding: 0.5rem 1.5rem !important;
        border-radius: 12px !important;
        font-weight: 600 !important;
        width: 100%;
        min-height: 3.5rem;
        margin-top: 10px;
    }

    /* Input Slide Specifics */
    .input-slide-container {
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
        height: 100%;
        text-align: center;
    }
    
    /* Plotly Charts */
    .js-plotly-plot .plotly .modebar { display: none !important; }

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
if 'auto_play' not in st.session_state:
    st.session_state.auto_play = False

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
    with st.container(border=True):
        st.markdown('## 📝 Stage 1: Input & Tokenization')

        col1, col2 = st.columns([6, 4])
        with col1:
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
    with st.container(border=True):
        st.markdown('## 🧮 Stage 2: Embedding Vectors')

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
                        height=500, # Increased height for projector
                        margin=dict(l=40, r=20, t=50, b=40),
                        plot_bgcolor='white',
                        paper_bgcolor='white',
                        xaxis=dict(title="Dimension Index", showgrid=False),
                        yaxis=dict(title="Embedding Value", showgrid=True, gridcolor='#f3f4f6'),
                        font=dict(family="Inter", size=14)
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
    with st.container(border=True):
        st.markdown('## ⚡ Stage 3: Encoder Processing')

        col1, col2 = st.columns([7, 5])

        with col1:
            st.markdown("#### 🏗️ Deep Encoder Stack (6 Layers)")
            
            # Create a grid for layers to save vertical space
            # We want to visualize data flowing through layers
            
            layer_cols = st.columns(2)
            
            for i in range(result['architecture_info']['encoder_layers']):
                # Distribute layers across 2 columns
                current_col = layer_cols[i % 2]
                
                with current_col:
                    if i <= st.session_state.animation_idx:
                        # Active layer with blue background
                        st.markdown(textwrap.dedent(f"""
                        <div style="background: #2563eb; color: white; border: 2px solid #1d4ed8;
                        border-radius: 12px; padding: 1rem; margin: 0.5rem 0; text-align: center;
                        font-size: 1.1rem; font-weight: 700; box-shadow: 0 4px 6px -1px rgba(37, 99, 235, 0.4);">
                            Layer {i+1}<br><span style="font-size: 0.9rem; font-weight: 400; opacity: 0.9;">Self-Attention + FFN</span>
                        </div>
                        """), unsafe_allow_html=True)
                    else:
                        # Inactive layer with gray border
                        st.markdown(textwrap.dedent(f"""
                        <div style="background: white; color: #6b7280; border: 2px solid #e5e7eb;
                        border-radius: 12px; padding: 1rem; margin: 0.5rem 0; text-align: center;
                        font-size: 1.1rem; font-weight: 600;">
                            Layer {i+1}<br><span style="font-size: 0.9rem; font-weight: 400;">Self-Attention + FFN</span>
                        </div>
                        """), unsafe_allow_html=True)



    with col2:
        st.markdown("### 💡 The Encoder's Job")

        st.markdown(f"""
        The encoder transforms the input text into a rich numerical representation. It uses **{result['architecture_info']['encoder_layers']} stacked layers** to understand language nuances.
        """)

        st.markdown("**Inside Each Layer:**")
        st.markdown("""
        1.  **Multi-Head Self-Attention**:
            *   The model looks at *every word simultaneously*.
            *   It calculates relationships (e.g., "bank" + "river" vs "bank" + "money").
            *   **Key Insight**: This parallel processing is what makes Transformers faster than older RNNs.

        2.  **Add & Norm**:
            *   Adds the original input back (Residual Connection) to prevent forgetting.
            *   Normalizes data to keep training stable.

        3.  **Feed-Forward Network (FFN)**:
            *   A small neural network applied to each token independently.
            *   Processes the "meaning" extracted by attention.
        """)

        st.info("**Deep Learning Context**: Lower layers (1-2) often capture grammar/syntax, while higher layers (5-6) capture semantic meaning and context.")

def render_stage_3(result):
    """Attention"""
    with st.container(border=True):
        st.markdown('## 🎯 Stage 4: Attention Mechanism')

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
    with st.container(border=True):
        st.markdown('## ⚡ Stage 5: Output Generation')

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
                <div style="background: #f9fafb; border: 3px dashed #d1d5db; border-radius: 20px;
                padding: 2.5rem; margin: 1.5rem 0; font-size: 2.5rem; color: #1f2937; min-height: 120px;
                display: flex; align-items: center; justify-content: center; font-weight: 500;">
                    {html.escape(step["generated_so_far"]) or "[start]"}
                </div>
                """), unsafe_allow_html=True)

                st.markdown("<div style='text-align: center; font-size: 4rem; color: #9ca3af; margin: 1.5rem 0;'>↓</div>", unsafe_allow_html=True)

                # Next predicted token
                next_token_display = html.escape(step["next_token"])
                st.markdown("**Next token:**")
                st.markdown(textwrap.dedent(f"""
                <div style="background: linear-gradient(135deg, #2563eb, #1d4ed8); color: white;
                border-radius: 20px; padding: 2.5rem; margin: 1.5rem 0; font-size: 3.5rem;
                font-weight: 700; text-align: center; box-shadow: 0 10px 15px -3px rgba(37, 99, 235, 0.4);">
                    {next_token_display}
                </div>
                """), unsafe_allow_html=True)

                st.markdown("**Top predictions:**")
                for token, prob in step['top_predictions'][:3]:
                    st.markdown(f"• **{token}**: {prob:.1%}")



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
    with st.container(border=True):
        st.markdown('## 🎉 Stage 6: Final Translation')

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

                <p style='font-size: 1.25rem; color: #6b7280; margin: 1.5rem 0 0.5rem 0; font-weight: 600;'>Input (English):</p>
                <div style="margin-bottom: 2rem;">
                    {input_html}
                </div>

                <div class="flow-arrow" style="font-size: 3rem;">⬇</div>

                <p style='font-size: 1.25rem; color: #6b7280; margin: 1rem 0 0.5rem 0; font-weight: 600;'>Output (Tamil):</p>
                <div style="margin-bottom: 2rem;">
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

# Initialize Session State
if 'page_mode' not in st.session_state:
    st.session_state.page_mode = 'setup'
if 'current_stage' not in st.session_state:
    st.session_state.current_stage = 0
if 'animation_idx' not in st.session_state:
    st.session_state.animation_idx = 0

def get_stage_max_steps(stage_idx, result):
    """Return the number of animation steps for a given stage"""
    if stage_idx == 0: # Tokenization
        return len(result['input_tokens'])
    elif stage_idx == 1: # Embedding (Static view)
        return 1
    elif stage_idx == 2: # Encoder (Layers)
        return result['architecture_info']['encoder_layers']
    elif stage_idx == 3: # Attention (Per output token)
        return len(result['output_tokens'])
    elif stage_idx == 4: # Generation (Step by step)
        return len(result['step_by_step'])
    elif stage_idx == 5: # Final
        return 1
    return 1

# ==========================================
# MAIN APP LOGIC
# ==========================================

if st.session_state.page_mode == 'setup':
    # ------------------------------------------
    # SLIDE 0: SETUP & INPUT
    # ------------------------------------------
    
    with st.container(border=True):
        # Centering spacer
        st.markdown("<div style='height: 10vh;'></div>", unsafe_allow_html=True)
        st.markdown("<h1 style='text-align: center; margin-bottom: 1rem;'>🌐 Neural Translation Visualizer</h1>", unsafe_allow_html=True)
        st.markdown("<p style='font-size: 2rem; color: #64748b; margin-bottom: 3rem; text-align: center;'>Interactive AI Presentation: English ➡ Tamil</p>", unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            selected = st.selectbox("Choose an Example:", list(EXAMPLES.keys()))
            text_input = st.text_input("Or type your own:", value=EXAMPLES[selected])
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            if st.button("🚀 Start Presentation", type="primary", use_container_width=True):
                with st.spinner("Initializing AI Model..."):
                    try:
                        viz = load_model()
                        st.session_state.visualizer = viz
                        
                        # Perform translation
                        result = viz.translate_with_details(text_input)
                        st.session_state.translation_result = result
                        
                        st.session_state.page_mode = 'presentation'
                        st.session_state.current_stage = 0
                        st.session_state.animation_idx = 0
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error: {e}")
        
        st.markdown("</div>", unsafe_allow_html=True)

elif st.session_state.page_mode == 'presentation' and st.session_state.translation_result:
    # ------------------------------------------
    # PRESENTATION MODE
    # ------------------------------------------
    result = st.session_state.translation_result
    
    # 1. Render Current Stage Content
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
        
    # 2. Navigation Bar (Floating/Sticky)
    
    # Calculate progress
    max_steps = get_stage_max_steps(st.session_state.current_stage, result)
    total_stages = len(STAGES)
    
    # State checks
    is_last_slide = (st.session_state.current_stage == len(STAGES) - 1)
    is_last_step = (st.session_state.animation_idx >= max_steps - 1)
    
    # Custom Navigation UI
    col1, col2, col3, col4, col5 = st.columns([1, 1.5, 3, 1, 1])
    
    with col1:
        if st.button("◀ Back", use_container_width=True):
            st.session_state.auto_play = False # Stop auto-play on manual interaction
            if st.session_state.animation_idx > 0:
                st.session_state.animation_idx -= 1
                st.rerun()
            elif st.session_state.current_stage > 0:
                st.session_state.current_stage -= 1
                st.session_state.animation_idx = 0 # Start of prev slide
                st.rerun()
                
    with col2:
        if st.session_state.auto_play:
            if st.button("⏸ Pause", type="primary", use_container_width=True):
                st.session_state.auto_play = False
                st.rerun()
        else:
            if st.button("▶ Auto-Play", use_container_width=True):
                st.session_state.auto_play = True
                st.rerun()
                
    with col3:
        # Progress Info
        current_step = st.session_state.animation_idx + 1
        st.markdown(f"<div style='text-align: center; font-size: 1.2rem; padding-top: 10px; color: #64748b;'><b>{STAGES[st.session_state.current_stage]}</b> • Step {current_step}/{max_steps}</div>", unsafe_allow_html=True)

    with col4:
        # Smart "Next" Button
        if is_last_slide and is_last_step:
            if st.button("🔄 Restart", type="primary", use_container_width=True):
                st.session_state.page_mode = 'setup'
                st.session_state.auto_play = False
                st.rerun()
        else:
            if st.button("Next ▶", type="primary", use_container_width=True):
                st.session_state.auto_play = False # Stop auto-play on manual interaction
                if not is_last_step:
                    st.session_state.animation_idx += 1
                    st.rerun()
                elif not is_last_slide:
                    st.session_state.current_stage += 1
                    st.session_state.animation_idx = 0
                    st.rerun()

    with col5:
        if st.button("🏠", help="Home", use_container_width=True):
            st.session_state.page_mode = 'setup'
            st.session_state.auto_play = False
            st.rerun()

    # 3. Auto-Play Logic
    if st.session_state.auto_play:
        # Dynamic delay based on stage content
        delay = 2.0
        if st.session_state.current_stage == 4: # Generation stage - faster for tokens
            delay = 1.0
        
        time.sleep(delay)
        
        if not is_last_step:
            st.session_state.animation_idx += 1
            st.rerun()
        elif not is_last_slide:
            st.session_state.current_stage += 1
            st.session_state.animation_idx = 0
            st.rerun()
        else:
            st.session_state.auto_play = False
            st.rerun()

else:
    # Fallback / Error state
    st.session_state.page_mode = 'setup'
    st.rerun()
