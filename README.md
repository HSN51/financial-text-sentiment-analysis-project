# 📈 Financial Text Sentiment Analysis Project

Analyze financial news articles to understand market sentiment using the **FinBERT** model. This project fetches articles from Yahoo Finance RSS feeds, processes them for sentiment analysis, and provides insights through detailed reports and visualizations. Ideal for traders, data scientists, or anyone interested in finance!

---

## 🌟 Features

- **Automated Article Fetching**: Fetch the latest financial news from Yahoo Finance based on stock tickers or keywords.
- **Sentiment Analysis**: Classify financial articles as Positive, Negative, or Neutral using FinBERT.
- **Customizable Date Range**: Filter articles for specific time periods to focus on recent events or trends.
- **Visualization**: Generate a pie chart to summarize sentiment distribution.
- **Export Reports**: Save analysis results in CSV format for further review.

---

## 📊 Sample Outputs

### Sentiment Distribution Pie Chart
![Sentiment Distribution](sentiment_distribution.png)

### Detailed CSV Report Example
| Title                              | Published            | Link                                   | Summary                       | Sentiment | Score |
|------------------------------------|----------------------|----------------------------------------|--------------------------------|-----------|-------|
| Gold prices rise amid inflation...| Tue, 03 Dec 2024... | [Article Link](https://example.com)   | Gold prices soar in...        | Positive  | 0.85  |
| Oil prices drop on market fears... | Mon, 02 Dec 2024... | [Article Link](https://example.com)   | Oil markets react to...        | Negative  | 0.75  |

---

## 🛠️ Technologies Used

- **Python**: Core programming language for the project.
- **Hugging Face Transformers**: Implements the FinBERT model for sentiment analysis.
- **Feedparser**: Parses financial RSS feeds from Yahoo Finance.
- **Matplotlib**: Generates sentiment distribution visualizations.
- **Pandas**: Organizes and exports sentiment analysis results as CSV files.

---

## 🚀 How to Run

### Prerequisites:
- Python 3.8+ installed on your system
- Git installed (optional, for version control)

---

### Steps:
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/HSN51/financial-text-sentiment-analysis-project.git
   cd financial-text-sentiment-analysis-project
   pip install -r requirements.txt
   python main.py

### 📧 Contact:
- **Name**: Hasan Hüseyin Gümüştepe
- **Email**:hgumustepe1@gmail.com
