🛒 Flipkart Laptop Recommender System
This is an AI-powered laptop recommender system built using web scraping, data preprocessing, and a Flask-based web app. It leverages product data from Flipkart and recommends budget laptops using NLP, vector search (FAISS), and a GenAI assistant with Retrieval-Augmented Generation (RAG).

📁 Project Structure
bash
Copy
Edit
├── app.py                          # Flask web application
├── web_scraping.py                # Script to scrape laptop data from Flipkart
├── flipkart_laptop_cleaned.csv    # Original scraped and cleaned dataset
├── flipkart_laptop_cleaned_new.csv # Newly updated/processed dataset
├── Dockerfile.py                  # Dockerfile to containerize the application
├── requirement.txt                # List of dependencies
🚀 Features
💻 Web scraping of laptop listings from Flipkart

📊 Cleaned and preprocessed laptop dataset

🔍 NLP-powered semantic search for laptop recommendations

🧠 GenAI-based assistant for question answering (via RAG)

🌐 Flask-based UI for interaction

🐳 Docker support for easy deployment

🛠️ Installation
Prerequisites
Python 3.8+

Docker (optional, for containerized setup)

Install using pip
bash
Copy
Edit
git clone https://github.com/your-username/flipkart-laptop-recommender.git
cd flipkart-laptop-recommender
pip install -r requirement.txt
python app.py
Run with Docker
bash
Copy
Edit
docker build -f Dockerfile.py -t flipkart-laptop-app .
docker run -p 5000:5000 flipkart-laptop-app
📄 File Descriptions
File	Description
app.py	Main Flask app for serving the recommender and chatbot
web_scraping.py	Script to extract laptop data from Flipkart
flipkart_laptop_cleaned.csv	Original cleaned data from Flipkart
flipkart_laptop_cleaned_new.csv	New or updated version of the dataset
Dockerfile.py	Docker instructions to build and run the project
requirement.txt	Python libraries required to run the project

🧠 Technologies Used
Python

Flask

BeautifulSoup / Selenium

FAISS / HuggingFace

LangChain / Ollama

Pandas / NumPy

Docker

✨ Future Improvements
Add user reviews and ratings analysis

Integrate GPU-based vector similarity search

UI/UX enhancement

Login and personalization

