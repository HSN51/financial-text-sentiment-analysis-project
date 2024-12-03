import feedparser
from transformers import pipeline
import matplotlib.pyplot as plt
from datetime import datetime, timezone
import pandas as pd

# Function to generate a pie chart visualization for sentiment distribution
def plot_sentiment(sentiments):
    # Define labels and sizes for the pie chart based on sentiment counts
    labels = ['Positive', 'Negative', 'Neutral']
    sizes = [sentiments['positive'], sentiments['negative'], sentiments['neutral']]
    
    # Set up the figure and plot the pie chart
    plt.figure(figsize=(6, 6))
    plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140)
    plt.title('Sentiment Distribution')  # Title for the chart
    
    # Save the chart as a PNG file
    plt.savefig('sentiment_distribution.png')
    
    # Display the chart
    plt.show()

# Function to summarize and export results
def summarize_results(detailed_report, final_score):
    # Convert the detailed report into a DataFrame for easy manipulation and display
    df = pd.DataFrame(detailed_report)
    
    # Print the detailed report to the console
    print("\n--- Detailed Report ---\n")
    print(df)
    
    # Calculate and display the overall sentiment based on the final score
    print("\n--- Final Sentiment ---")
    overall_sentiment = (
        "Positive" if final_score >= 0.15 else
        "Negative" if final_score <= -0.15 else
        "Neutral"
    )
    print(f"Overall Sentiment: {overall_sentiment}, Score: {final_score:.2f}")
    
    # Save the detailed report as a CSV file
    df.to_csv('sentiment_report.csv', index=False)

# Define the stock ticker symbol and keyword to filter articles
ticker = 'GC=F'  # Example: Gold futures
keyword = 'gold'  # Keyword for filtering relevant articles

# Load the FinBERT model for sentiment analysis from Hugging Face
pipe = pipeline("text-classification", model="ProsusAI/finbert")

# Define the RSS feed URL for fetching financial news
res_url = f'https://finance.yahoo.com/rss/headline?s={ticker}'
feed = feedparser.parse(res_url)  # Parse the RSS feed

# Initialize variables to store total sentiment score and article count
total_score = 0
num_articles = 0

# Dictionary to keep track of sentiment counts for visualization
sentiment_counts = {'positive': 0, 'negative': 0, 'neutral': 0}

# List to store detailed results for each article
detailed_report = []

# Define the date range for filtering articles
start_date = datetime(2024, 11, 3, tzinfo=timezone.utc)  
end_date = datetime(2024, 12, 3, tzinfo=timezone.utc)    

# Iterate through each article in the RSS feed
for i, entry in enumerate(feed.entries):
    # Convert the published date of the article to a datetime object
    published_date = datetime.strptime(entry.published, '%a, %d %b %Y %H:%M:%S %z')
    
    # Skip articles outside the defined date range
    if not (start_date <= published_date <= end_date):
        continue

    # Skip articles that do not contain the specified keyword in their summary
    if keyword.lower() not in entry.summary.lower():
        continue

    # Perform sentiment analysis on the article summary using the FinBERT model
    sentiment = pipe(entry.summary)[0]
    
    # Update the sentiment counts for visualization
    sentiment_counts[sentiment['label'].lower()] += 1
    
    # Update the total score based on the sentiment label
    if sentiment['label'] == 'positive':
        total_score += sentiment['score']
    elif sentiment['label'] == 'negative':
        total_score -= sentiment['score']
    
    # Increment the article count
    num_articles += 1

    # Append detailed information about the article to the report list
    detailed_report.append({
        'title': entry.title,         
        'published': entry.published, 
        'link': entry.link,           
        'summary': entry.summary,     
        'sentiment': sentiment['label'], # Sentiment label (positive/negative/neutral)
        'score': sentiment['score']      
    })

# Generate a pie chart for sentiment distribution
plot_sentiment(sentiment_counts)

# Calculate the final sentiment score (average sentiment score across articles)
final_score = total_score / num_articles if num_articles > 0 else 0

# Summarize and export the results
summarize_results(detailed_report, final_score)
