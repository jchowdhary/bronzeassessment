# GenAI Bronze Assessment - Machine Learning & Generative AI Portfolio

This repository contains a comprehensive collection of Jupyter notebooks demonstrating both traditional machine learning techniques and cutting-edge Generative AI applications. The project is organized into two layers: **bronzeLayer** (classical ML) and **silverLayer** (GenAI applications).

## 📁 Repository Structure

```
genAIBronzeAssessment/
├── bronzeLayer/              # Traditional Machine Learning Projects
│   ├── dataset/              # Training datasets
│   ├── assessmentquestions.txt
│   └── *.ipynb               # ML notebooks
├── silverLayer/              # Generative AI Applications
│   ├── dataset/              # GenAI datasets (PDFs, etc.)
│   ├── audio_database/       # Audio files for similarity search
│   ├── chroma_db/           # Vector database storage
│   ├── assessmentquestion.txt
│   └── *.ipynb               # GenAI notebooks
├── requirements.txt          # Python dependencies
└── README.md                # This file
```

---

## 🥉 Bronze Layer - Traditional Machine Learning

### Overview
The bronzeLayer contains classical machine learning demonstrations covering regression, classification, and neural networks.

### 📓 Available Notebooks

1. **neuralnetwork_with_kerastensorflow.ipynb**
   - Deep neural network implementation using Keras/TensorFlow
   - Model architecture visualization
   - Classification tasks

2. **employee_attriction_knn_classification.ipynb**
   - Employee attrition prediction using K-Nearest Neighbors
   - Feature engineering and preprocessing
   - Classification metrics evaluation

3. **housing_regression_3dplot.ipynb**
   - Housing price prediction with regression models
   - 3D visualization of feature relationships
   - Advanced plotting techniques

4. **multivariate_linear_regression.ipynb**
   - Multiple linear regression implementation
   - Feature selection and analysis
   - Model evaluation metrics

5. **AWS_Cost_Prediction_Complete.ipynb** & **AWS_Cost_Prediction_Universal.ipynb**
   - Cloud cost prediction models
   - FinOps machine learning applications
   - Time series forecasting

### 🎯 Technologies Used (Bronze Layer)
- **ML Frameworks**: scikit-learn, TensorFlow, Keras
- **Data Processing**: pandas, numpy
- **Visualization**: matplotlib, seaborn, plotly
- **Model Persistence**: joblib
- **Other**: xgboost, scipy, pydot, graphviz

---

## 🥈 Silver Layer - Generative AI Applications

### Overview
The silverLayer showcases advanced Generative AI applications using state-of-the-art LLMs, vector databases, and multimodal models.

### 📓 Available Notebooks

#### 1. **assignment1_pdf_summarization.ipynb**
**PDF Summarization using RAG with Open Source LLM**

- **Objective**: Intelligent document summarization using Retrieval Augmented Generation
- **Technologies**:
  - LangChain for orchestration
  - ChromaDB vector database
  - Groq API with Llama 3.3-70b model
  - HuggingFace embeddings (sentence-transformers/all-MiniLM-L6-v2)
  - PyPDF for document processing
- **Features**:
  - PDF text extraction and chunking
  - Vector embedding generation and storage
  - Context-aware question answering
  - Document summarization

#### 2. **assignment2_audiosimilarity_search.ipynb**
**Audio Similarity Search**

- **Objective**: Compare audio files using deep learning embeddings
- **Technologies**:
  - Wav2Vec2 pre-trained model (Facebook AI)
  - torchaudio for audio processing
  - scipy for similarity metrics
- **Features**:
  - Audio embedding extraction (768-dimensional)
  - Cosine similarity computation
  - Comparative analysis with visualization
  - Support for multiple audio formats (mp3, wav, flac)

#### 3. **assignment3_genai_speech_translator.ipynb**
**GenAI-Powered Speech-to-Speech Translator**

- **Objective**: Intelligent multilingual speech translation system
- **Technologies**:
  - OpenAI Whisper (speech recognition)
  - Groq (Llama) / Google Gemini (translation)
  - Edge TTS (speech synthesis)
- **Features**:
  - Multi-language support (15+ languages)
  - Translation styles: standard, formal, casual, poetic, technical, simplified
  - Context-aware translations with conversation memory
  - Natural voice synthesis
  - Translation explanations for learning
- **Supported Languages**: English, Spanish, French, German, Hindi, Chinese, Japanese, Korean, Arabic, Portuguese, Russian, Italian, Dutch, Polish, Turkish

#### 4. **assignment4_mongodb.ipynb**
**MongoDB Integration with GenAI**

- Document database operations
- Vector search capabilities
- Integration with LangChain

#### 5. **capstone1_assignment.ipynb**
**Capstone Project**

- Comprehensive GenAI application
- Combines multiple technologies
- End-to-end solution demonstration

### 🎯 Technologies Used (Silver Layer)
- **LLM Frameworks**: LangChain, LangChain-Core, LangChain-Community
- **LLM Providers**: Groq (Llama), Google Gemini
- **Vector Databases**: ChromaDB, FAISS
- **Embeddings**: HuggingFace (sentence-transformers), OpenAI
- **Document Processing**: PyPDF, PyMuPDF
- **Audio**: torch, torchaudio, Whisper, Edge TTS, pydub
- **Database**: MongoDB (pymongo)
- **Utilities**: python-dotenv, tiktoken, pydantic

---

## 🚀 Getting Started

### Prerequisites
- Python 3.10 or 3.11 (recommended)
- Conda or venv for environment management
- Git

### 1. Clone the Repository

```bash
git clone https://github.com/<your-username>/genAIBronzeAssessment.git
cd genAIBronzeAssessment
```

Replace `<your-username>` with your GitHub username.

### 2. Environment Setup

You can create separate environments for bronze and silver layers, or use a single environment for both.

#### Option A: Single Environment (Recommended)

```bash
# Create a conda environment
conda create -n genai python=3.11.11 -y

# Activate the environment
conda activate genai

# Install all dependencies
pip install -r requirements.txt
```

#### Option B: Separate Environments

**For Bronze Layer:**
```bash
conda create -n bronze python=3.11.11 -y
conda activate bronze
# Install only bronze layer packages (uncomment relevant lines in requirements.txt)
```

**For Silver Layer:**
```bash
conda create -n llms python=3.11.11 -y
conda activate llms
# Install only silver layer packages (uncomment relevant lines in requirements.txt)
```

### 3. API Keys Configuration (Silver Layer Only)

For GenAI applications, you need API keys. Create a `.env` file in the root directory:

```bash
# .env file
GROQ_API_KEY=your_groq_api_key_here
GOOGLE_API_KEY=your_google_api_key_here
```

**Get your FREE API keys:**
- **Groq**: https://console.groq.com/keys
- **Google Gemini**: https://aistudio.google.com/app/apikey

### 4. Run the Notebooks

#### Using Jupyter Lab (Recommended)

```bash
# Start Jupyter Lab
jupyter lab

# Or Jupyter Notebook
jupyter notebook
```

#### Using VS Code

1. Install the Jupyter extension in VS Code
2. Open any `.ipynb` file
3. Select the appropriate kernel (`genai`, `bronze`, or `llms`)
4. Run cells interactively

---

## 📚 Usage Examples

### Bronze Layer Example

```python
# Running a classification model
# Open: bronzeLayer/employee_attriction_knn_classification.ipynb
# Execute all cells to train and evaluate the model
```

### Silver Layer Example

```python
# PDF Summarization with RAG
# 1. Place your PDF in silverLayer/dataset/
# 2. Open: silverLayer/assignment1_pdf_summarization.ipynb
# 3. Update the PDF path in the notebook
# 4. Run all cells to generate summaries
```

---

## 📊 Datasets

### Bronze Layer Datasets
- Employee attrition data
- Housing prices dataset
- AWS cost data
- Various CSV files for regression/classification

Location: `bronzeLayer/dataset/`

### Silver Layer Datasets
- Sample PDF documents for summarization
- Audio files for similarity search
- MongoDB document samples

Location: `silverLayer/dataset/` and `silverLayer/audio_database/`

---

## 🎓 Assessment Questions

- **Bronze Layer**: `bronzeLayer/assessmentquestions.txt`
- **Silver Layer**: `silverLayer/assessmentquestion.txt`

These files contain questions and exercises related to each layer's content.

---

## 🔧 Troubleshooting

### Common Issues

1. **Package Installation Errors**
   ```bash
   # Try upgrading pip first
   pip install --upgrade pip
   
   # Install packages one by one if batch install fails
   pip install langchain chromadb pypdf
   ```

2. **API Key Issues**
   - Ensure `.env` file is in the root directory
   - Verify API keys are valid and active
   - Check API rate limits

3. **Audio Processing Issues**
   ```bash
   # Install system dependencies for audio
   sudo apt-get install ffmpeg libsndfile1
   ```

4. **CUDA/GPU Issues**
   ```bash
   # Use CPU-only versions
   pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
   ```

5. **ChromaDB Persistence Issues**
   - Delete `silverLayer/chroma_db/` directory and recreate
   - Ensure write permissions in the directory

---

## 🌟 Key Features

### Bronze Layer Highlights
- ✅ Traditional ML algorithms (KNN, Linear Regression, Neural Networks)
- ✅ Comprehensive data visualization
- ✅ Model evaluation and metrics
- ✅ Real-world datasets (employee data, housing, AWS costs)

### Silver Layer Highlights
- ✅ State-of-the-art LLMs (Llama 3.3, Gemini 2.0)
- ✅ RAG implementation with vector databases
- ✅ Multimodal AI (audio processing, speech translation)
- ✅ Context-aware translations with style adaptation
- ✅ Free tier APIs (Groq, Google Gemini)
- ✅ Production-ready code patterns

---

## 🤝 Contributing

Feel free to submit issues and enhancement requests!

---

## 📝 License

This project is for educational purposes.

---

## 👥 Authors

- Assessment prepared for GenAI Course
- Bronze Layer: Traditional ML fundamentals
- Silver Layer: Advanced GenAI applications

---

## 📞 Support

For questions or issues:
- Check assessment question files in each layer
- Review notebook comments and documentation
- Refer to official documentation for each technology

---

## 🔄 Version Information

- **Python**: 3.10 or 3.11
- **LangChain**: Latest stable version
- **Groq Models**: Llama 3.3-70b-versatile (recommended)
- **Google Models**: Gemini 2.0 Flash (latest)
- **Whisper**: base model (good balance of speed/accuracy)

---

**Happy Learning! 🚀**
