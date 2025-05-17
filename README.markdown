# Flipkart Laptop Search Engine

## Project Overview
The **Flipkart Laptop Search Engine** is an AI-powered web application that recommends laptops based on user queries, such as price range, specifications, or use cases (e.g., "laptops under 40000 Rs" or "gaming laptop with 8GB RAM"). It uses **Retrieval-Augmented Generation (RAG)** to retrieve relevant laptops from a dataset and generate tailored recommendations. The application is built with **Python**, served via a **Gradio** web interface, and deployed on an **AWS EC2 instance** using **Docker**. The dataset is sourced from Flipkart and processed for efficient querying.

## Features
- **Query-Based Recommendations**: Supports natural language queries for laptops (e.g., "budget laptop for students").
- **RAG Pipeline**: Combines vector search (FAISS) with LLM-based reasoning (Groq) for accurate results.
- **Gradio Interface**: User-friendly web UI hosted on EC2.
- **Data Scraping**: Includes a script (`web_scraping.py`) to collect laptop data from Flipkart.
- **Filtered Search**: Handles queries with price, RAM, storage, and processor specifications.

## Tech Stack
- **Language**: Python 3.9
- **Web Framework**: Gradio
- **AI Libraries**:
  - LangChain (RAG pipeline)
  - LangChain-Groq (LLM)
  - LangChain-HuggingFace (embeddings)
  - FAISS (vector search)
- **Data Processing**: Pandas, regex
- **Deployment**: Docker, AWS EC2
- **Dependencies**: Listed in `requirement.txt` (e.g., `pandas`, `gradio`, `langchain`, `faiss-cpu`)
- **Datasets**: `flipkart_laptop_cleaned.csv`, `flipkart_laptop_cleaned_new.csv`

## Project Structure
```
Flipkart_Laptop_search_engine/
├── app.py                      # Main application (Gradio UI, RAG pipeline)
├── flipkart_laptop_cleaned.csv # Initial laptop dataset
├── flipkart_laptop_cleaned_new.csv # Updated laptop dataset
├── web_scraping.py             # Script to scrape laptop data from Flipkart
├── requirement.txt             # Python dependencies
├── Dockerfile                  # Docker configuration
├── .env                        # Environment variables (not in repo, for GROQ_API_KEY)
└── README.md                   # Project documentation
```

## Prerequisites
- **Docker**: For building and running the container.
- **AWS EC2**: Instance with ports 22 (SSH) and 7860 (Gradio) open.
- **Groq API Key**: From [Groq Console](https://console.groq.com/).
- **Git**: To clone the repository.
- **Docker Hub Account**: (Optional) For pushing the image.

## Setup Instructions

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/Flipkart_Laptop_search_engine.git
cd Flipkart_Laptop_search_engine
```
Replace `yourusername` with your GitHub username.

### 2. Configure Environment Variables
Create a `.env` file with your Groq API key:
```bash
echo "GROQ_API_KEY=your-groq-api-key" > .env
```
Or pass the key via `docker run` (recommended).

### 3. Build the Docker Image
```bash
docker build -t laptop-rag-app .
```

### 4. (Optional) Push to Docker Hub
```bash
docker login
docker tag laptop-rag-app:latest yourusername/laptop-rag-app:latest
docker push yourusername/laptop-rag-app:latest
```
Replace `yourusername` with your Docker Hub username.

### 5. Deploy on AWS EC2
1. **Launch EC2 Instance**:
   - Use Ubuntu 22.04 LTS, `t3.medium`, 20 GB storage.
   - Security group: Allow ports 22 (SSH) and 7860 (Gradio).
2. **SSH into EC2**:
   ```bash
   ssh -i my-key.pem ubuntu@<ec2-public-ip>
   ```
3. **Install Docker**:
   ```bash
   sudo apt-get update
   sudo apt-get install -y docker.io
   sudo usermod -aG docker ubuntu
   exit
   ssh -i my-key.pem ubuntu@<ec2-public-ip>
   ```
4. **Run Container**:
   ```bash
   docker run -d --restart=always -p 7860:7860 \
       --env GROQ_API_KEY=your-groq-api-key \
       laptop-rag-app:latest
   ```

### 6. Access the App
- Open `http://<ec2-public-ip>:7860` in a browser.
- Test queries like “laptops under 40000 Rs”.

## Usage
- **Web Interface**: Enter queries in the Gradio UI (e.g., “budget laptop for coding”).
- **Example Queries**:
  - “Laptops under 40000 Rs with 8GB RAM”
  - “Gaming laptop around 1 lakh”
  - “Compare laptops with SSD”
- **Data Scraping**: Run `web_scraping.py` to update the dataset (requires additional setup).

## Troubleshooting
- **401 Invalid API Key**:
  - Verify `GROQ_API_KEY` in `docker run` or `.env`.
  - Test key:
    ```bash
    curl -X POST "https://api.groq.com/openai/v1/chat/completions" \
         -H "Authorization: Bearer your-groq-api-key" \
         -H "Content-Type: application/json" \
         -d '{"model": "llama-3.1-8b-instant", "messages": [{"role": "user", "content": "Hello"}]}'
    ```
- **UI Not Loading**:
  - Check security group (port 7860).
  - Verify container: `docker ps`.
  - Inspect logs: `docker logs <container-id>`.

## License
MIT License (or specify your preferred license).

## Contact
Open an issue on [GitHub](https://github.com/yourusername/Flipkart_Laptop_search_engine) or email [your-email@example.com].