# 🌐 NLP Translation Visualizer

An interactive web application that visualizes how transformer-based neural machine translation works, built with Streamlit.

## 🎯 Features

This application provides comprehensive visualizations of the translation process:

### 1. 🔤 Tokenization Visualization
- See how text is broken down into tokens
- View token IDs and understand the tokenization process
- Compare input (English) and output (Tamil) tokens side-by-side

### 2. 🏗️ Architecture Diagram
- Interactive view of the transformer encoder-decoder architecture
- Visual representation of data flow through the model
- Detailed information about layers, dimensions, and model parameters

### 3. 🎯 Attention Heatmap
- Visualize cross-attention between input and output tokens
- See which words the model focuses on during translation
- Interactive heatmap with hover details

### 4. ⚡ Translation Flow Animation
- Step-by-step token generation
- View probability distributions for each prediction
- Interactive playback controls (play, pause, step forward/backward)
- Auto-play mode for presentations

### 5. 📐 Math Explained
- Simplified explanations of underlying mathematical operations
- Visual representation of:
  - Token embeddings
  - Attention mechanism
  - Softmax function
- Interactive charts showing actual values from the model

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- pip (Python package installer)
- At least 4GB of RAM (for model loading)

### Installation

1. Clone or download this repository

2. Install dependencies:
```bash
pip install -r requirements.txt
```

The installation will download:
- Streamlit for the web interface
- Transformers library from HuggingFace
- PyTorch for running the model
- Plotly for interactive visualizations

### Running the Application

1. Navigate to the project directory:
```bash
cd TranslatorNLP
```

2. Run the Streamlit app:
```bash
streamlit run app.py
```

3. The application will automatically open in your default web browser at `http://localhost:8501`

4. **First time usage**: The model will download automatically (approximately 300MB). This may take a few minutes depending on your internet connection.

## 📖 How to Use

### Basic Usage

1. **Select an example** from the sidebar dropdown, or enter your own text
2. Make sure to include `>>tam<<` at the beginning of your text to specify Tamil as the target language
3. Click **"🚀 Translate & Visualize"**
4. Explore the five visualization tabs

### Preset Examples

The application includes 5 preset examples:
- Simple sentence
- Government announcement (default)
- Question
- Sentence with numbers
- Complex sentence

### Animation Controls

In the "Translation Flow" tab:
- **⏮️ Start**: Jump to the first step
- **◀️ Prev**: Go to previous step
- **▶️ Next**: Go to next step
- **⏭️ End**: Jump to the last step
- **Auto-play**: Enable continuous animation
- **Animation Speed**: Adjust in the sidebar (0.1x to 2.0x)

### Tips for Best Results

- Keep input sentences under 50 words for faster processing
- Always include `>>tam<<` at the beginning for Tamil translation
- Use the preset examples to see well-formatted inputs
- Enable auto-play in the Translation Flow tab for demonstrations

## 🎬 Presentation Mode

For presentations:

1. Press **F11** in your browser for fullscreen mode
2. Use preset examples for quick demonstrations
3. Adjust animation speed in the sidebar based on audience
4. Start with the Tokenization tab and progress through to Math Explained

### Recommended Demo Flow (2-3 minutes)

1. **Introduction** (30s): Show the main interface and explain the goal
2. **Tokenization** (30s): Demonstrate how text is split into tokens
3. **Architecture** (30s): Quick overview of the encoder-decoder structure
4. **Attention** (45s): Show the attention heatmap and explain what it means
5. **Live Demo** (45s): Take audience input and translate it

## 📁 Project Structure

```
TranslatorNLP/
├── app.py                 # Main Streamlit application
├── model_wrapper.py       # Translation model wrapper
├── main.py                # Original simple script (reference)
├── requirements.txt       # Python dependencies
├── .streamlit/
│   └── config.toml       # Streamlit theme configuration
└── README.md             # This file
```

## 🔧 Technical Details

### Model Information

- **Model**: Helsinki-NLP/opus-mt-en-dra
- **Architecture**: MarianMT (Transformer-based)
- **Source Language**: English
- **Target Languages**: Dravidian languages (Tamil, Telugu, Malayalam, etc.)
- **Model Size**: ~300MB

### How It Works

1. **Tokenization**: Input text is split into subword tokens using SentencePiece
2. **Embedding**: Tokens are converted to dense vector representations
3. **Encoding**: The encoder processes input tokens through multiple transformer layers
4. **Decoding**: The decoder generates output tokens autoregressively
5. **Attention**: Cross-attention mechanism helps the decoder focus on relevant input tokens

### Performance Notes

- Model loads in ~20-30 seconds on first run (cached afterwards)
- Translation typically takes 1-3 seconds
- Visualization rendering is near-instant

## 🐛 Troubleshooting

### Model fails to load
- **Solution**: Make sure you have at least 4GB of free RAM
- Try restarting the application
- Check your internet connection (for first-time download)

### Slow performance
- **Solution**: Reduce input text length (keep under 50 words)
- Close other heavy applications
- Consider using a machine with more RAM

### Import errors
- **Solution**: Reinstall dependencies:
  ```bash
  pip install --upgrade -r requirements.txt
  ```

### Port already in use
- **Solution**: Streamlit will automatically try the next available port, or specify one:
  ```bash
  streamlit run app.py --server.port 8502
  ```

## 🤝 Contributing

This is a presentation/educational tool. Feel free to:
- Add support for other translation models
- Improve visualizations
- Add more preset examples
- Enhance the UI/UX

## 📄 License

This project uses the following open-source components:
- HuggingFace Transformers (Apache 2.0)
- Helsinki-NLP OPUS-MT models (Apache 2.0)
- Streamlit (Apache 2.0)

## 🙏 Acknowledgments

- Helsinki-NLP for the OPUS-MT translation models
- HuggingFace for the Transformers library
- Streamlit team for the amazing framework

## 📞 Support

For issues or questions:
1. Check the Troubleshooting section above
2. Review the [Streamlit documentation](https://docs.streamlit.io/)
3. Check the [HuggingFace documentation](https://huggingface.co/docs)

---

**Built with ❤️ for educational purposes**

*This visualization demonstrates transformer-based neural machine translation in an accessible, interactive format suitable for presentations and teaching.*
