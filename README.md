# Hotel-Reviews Sentiment Analysis 🌟🏨

## 📖 Project Overview
**Hotel-Reviews Sentiment Analysis** aims to identify the sentiment of hotel reviews left by customers. The project leverages various techniques from Natural Language Processing (NLP), Machine Learning, and Data Visualization to understand the emotions behind guest feedback. 

## 🚀 Features
- **Sentiment Prediction** using Machine Learning (Random Forest Classifier) 🌲
- **Text Preprocessing** and Cleaning using NLP Techniques 📝
- **Feature Extraction** using TF-IDF Vectorization 📊
- **Embedding Creation** using Doc2Vec 🧠
- **Exploratory Data Analysis (EDA)** to derive meaningful insights 📈
- **Visualizations** to interpret review sentiment trends 🎨

## 🔧 Techniques and Tools
1. **Data Cleaning and Preprocessing** ✨
   - Removal of stop words, punctuation, and special characters.
   - Lemmatization for standardizing words to their root form.

2. **Exploratory Data Analysis (EDA)** 🧐
   - Visualizations to explore the distribution of sentiments in the reviews.
   - Analysis of review length, word frequency, and common keywords.

3. **NLP Techniques** for Feature Extraction 🧰
   - **Doc2Vec Embeddings** for capturing semantic relationships in reviews.
   - **TF-IDF Vectorization** for quantifying the importance of words.

4. **Machine Learning Model** for Sentiment Prediction 💡
   - **Random Forest Classifier** to predict sentiments (Positive/Negative/Neutral).
   - Evaluation metrics to measure model performance.

## 📊 Visualizations
- **Word Clouds** to showcase frequently used words 🌤️
- **Sentiment Distribution Charts** to analyze the spread of sentiments 📉
- **Correlation Heatmaps** to explore relationships between features 🔥

## 📂 Project Structure
```bash
Hotel-Reviews-Sentiment-Analysis/
│
├── data/                    # Dataset and processed files
├── notebooks/               # Jupyter notebooks for EDA, NLP, and Modeling
├── models/                  # Trained ML models
├── app/                     # Streamlit app code
└── README.md                # Project documentation
## 💻 Usage
1. **Clone the repository**:
   ```bash
   git clone https://github.com/Naveen035/Hotel-Reviews-Sentiment-Analysis.git
   cd Hotel-Reviews-Sentiment-Analysis
2. **Install required dependencies**:
```bash
  pip install -r requirements.txt
3. **Run the Streamlit app**:
```bash
  streamlit run app/app.py
## 🏆 Results
The project accurately predicts the sentiment of hotel reviews and provides valuable insights into customer feedback patterns. Visualizations help in understanding sentiment trends and identifying key words and phrases in guest reviews.
