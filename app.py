import os
import re
import pandas as pd
import gradio as gr # Assuming this is for the UI part
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda, RunnableSequence
from dotenv import load_dotenv

# Load environment variables (ensure this part is at the top level of your script)
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY not found in .env file")

# Data Processing (standardize_field, process_data - unchanged from your provided code)
# These functions prepare the data that will be used by the RAG system (Step 3 in diagram)
def standardize_field(value, field_type, specs=None):
    """Standardize field values, parsing Specifications for missing Processor."""
    if pd.isna(value) or value == "Not found":
        if field_type == "processor" and specs:
            match = re.search(r'(Intel Core [^\s,]+|AMD Ryzen [^\s,]+)', specs, re.IGNORECASE)
            return match.group(0).title() if match else None
        return None
    value = str(value).strip()
    
    if field_type == "ram":
        match = re.match(r"(\d+)(?:\s*GB)?(?:\s*GB)?", value, re.IGNORECASE)
        return int(match.group(1)) if match else None
    elif field_type == "storage":
        match = re.match(r"(\d+)\s*(GB|TB)\s*(SSD|HDD)?", value, re.IGNORECASE)
        if match:
            size, unit, type_ = match.groups()
            size = int(size) * (1024 if unit.lower() == "tb" else 1)
            return f"{size}GB {type_ or 'SSD'}"
        return None
    elif field_type == "price": # Assuming prices are in Rupees as per your data
        cleaned_value = str(value).replace("₹", "").replace(",", "").strip()
        return float(cleaned_value) if cleaned_value else None
    elif field_type == "weight":
        match = re.match(r"(\d+\.?\d*)\s*(kg|Kg|KG)?", value, re.IGNORECASE)
        return float(match.group(1)) if match else None
    elif field_type == "rating":
        try:
            return float(value) if value != "Not found" else None
        except ValueError:
            return None
    elif field_type == "processor":
        return value.title() if value != "Not found" else None
    return value

def process_data(csv_file):
    """Load and process laptop data from CSV."""
    if not os.path.exists(csv_file):
        raise ValueError(f"CSV file not found: {csv_file}")
    try:
        df = pd.read_csv(csv_file)
    except Exception as e:
        raise ValueError(f"Error reading CSV file: {str(e)}")
    
    df["complete_count"] = df[["Processor", "RAM", "Storage", "Rating", "Weight"]].notna().sum(axis=1)
    df = df.sort_values("complete_count", ascending=False).drop_duplicates(
        subset=["Product Name", "Price", "RAM", "Storage"], keep="first"
    ).drop(columns="complete_count")
    
    documents = []
    for _, row in df.iterrows():
        metadata = {
            "product_name": row["Product Name"],
            "brand": row["Brand"],
            "price": standardize_field(row["Price"], "price"),
            "ram": standardize_field(row["RAM"], "ram"),
            "storage": standardize_field(row["Storage"], "storage"),
            "processor": standardize_field(row["Processor"], "processor", row["Specifications"]),
            "weight": standardize_field(row["Weight"], "weight"),
            "rating": standardize_field(row["Rating"], "rating"),
            "display_size": row["Display Size"],
            "os": row["OS"],
            "warranty": row["Warranty"],
        }
        
        content = (
            f"{row['Product Name']} | "
            f"Processor: {metadata['processor'] or 'Unknown'}, "
            f"RAM: {metadata['ram'] or 'Unknown'}GB, "
            f"Storage: {metadata['storage'] or 'Unknown'}, "
            f"Price: ₹{metadata['price'] or 'Unknown'}, " # Using Rupee symbol
            f"Brand: {metadata['brand']}, "
            f"Weight: {metadata['weight'] or 'Unknown'}kg, "
            f"Rating: {metadata['rating'] or 'Unknown'}"
        )
        
        doc = Document(page_content=content, metadata=metadata)
        documents.append(doc)
    
    return documents

# RAG Setup
def setup_rag(chunks):
    """Set up the RAG pipeline for laptop recommendations."""
    if not chunks or not isinstance(chunks, list):
        raise ValueError("Invalid or empty document chunks provided.")

    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    faiss_index_path = "faiss_index_laptops"
    if os.path.exists(faiss_index_path):
        try:
            vectorstore = FAISS.load_local(faiss_index_path, embedding_model, allow_dangerous_deserialization=True)
        except Exception as e:
            print(f"Warning: Failed to load FAISS index: {str(e)}. Rebuilding index.")
            vectorstore = FAISS.from_documents(chunks, embedding_model)
    else:
        vectorstore = FAISS.from_documents(chunks, embedding_model)
    
    vectorstore.save_local(faiss_index_path)
    
    try:
        llm = ChatGroq(
            model="llama-3.1-8b-instant",
            groq_api_key=GROQ_API_KEY,
            temperature=0.1,
            max_tokens=1500 # Sufficient for structured responses
        )
    except Exception as e:
        raise ValueError(f"Error initializing Groq model: {str(e)}")
    
    # MODIFIED TEMPLATE TO REFLECT THE 5-STEP WORKFLOW
    template = """
    You are an expert AI assistant for laptop recommendations. Your goal is to follow a structured thought process to provide the best recommendation based on the user's query.

    **Your Thought Process (Follow these steps internally for each new query that isn't a direct comparison follow-up):**

    1.  **Understand the Query (Corresponds to Diagram Step 2: LLM Parse query):**
        * Analyze the user's current query (`{question}`), considering the ongoing `Chat History` for context.
        * Identify the core needs: What category of product (e.g., laptop)? Any price constraints (e.g., "budget," "under 50000 rupees")? What are the key stated or inferred features (e.g., from "remote work," you might infer needs like "good battery," "lightweight," "reliable performance")? What is the primary use case (e.g., remote work, student, gaming)?
        * *Internal thought example for "Budget laptop for remote work": Needs a laptop, affordable (e.g., within a typical budget range for such a request, or as specified), good battery life, portable/lightweight, for professional remote work.*

    2.  **Identify Relevant Laptops from Provided Context (Corresponds to Diagram Step 3: RAG Retrieve products):**
        * Scan the provided `{context}` (which contains details of several laptops) for products that closely match the understood needs from Step 1. The `{context}` has already been filtered to be relevant to the user's query.

    3.  **Select the Best Match and its Most Relevant Feature (Corresponds to Diagram Step 4: AI Agent):**
        * From the relevant laptops found in `{context}`, pick the one that appears to be the single best fit for the user's primary requirements. Consider factors like price, suitability for the use case, and specific features mentioned or inferred.
        * Identify its most compelling feature(s) that directly address the core requirements.
        * *Internal thought example: "From the context, Acer Aspire 5 (P001) seems like a strong budget option for remote work. Key feature: Its 10-hour battery life is excellent for working on the go."*

    4.  **Formulate and Generate the Recommendation Description (Corresponds to Diagram Step 5: GenAI Generate description):**
        * Based on your selection in Step 3, generate a concise, user-friendly recommendation.
        * Start by directly recommending the chosen laptop.
        * Highlight 2-3 key attributes (like "affordable," "lightweight," specific spec) that make it suitable, including the compelling feature you identified. Frame it positively and tie it back to the user's query.
        * Example structure for the main recommendation: "For your need of a [user's stated need, e.g., budget laptop for remote work], the [Selected Laptop Name] is an excellent choice. It's [Attribute 1, e.g., affordable], [Attribute 2, e.g., lightweight], and features a [Key Feature Value, e.g., long 10-hour battery life], making it ideal for [use case]."
        * After presenting this primary recommendation, if there are other strong candidates from the `{context}` that also meet the primary criteria well, you can briefly mention one or two of them as "Other good options to consider are: ...".

    **Specific Handling for Comparison Follow-ups:**
    - If the user's current `Question` is a follow-up asking to "compare laptops" (e.g., "compare them," "compare all of them," "which is better between X and Y," "how do these differ?"), AND your IMMEDIATELY PRECEDING response in the `Chat History` listed specific laptops:
        - You MUST focus your comparison on THOSE SPECIFIC LAPTOPS FROM YOUR PREVIOUS RESPONSE.
        - Identify these laptops from the `Chat History`.
        - The `Context` provided for the current turn might not be the most relevant for this specific comparison; PRIORITIZE the laptops from your previous response.
        - Present a comparison of ONLY the identified laptops (e.g., in a table or side-by-side bullet points).
        - Follow with a brief analysis of their main differences, pros, and cons RELATIVE TO EACH OTHER.
        - Do NOT perform a new broad search when the user is clearly asking to compare items from the immediate chat progression. If unclear, ask for clarification.

    **General Response Guidelines:**
    - For initial recommendations (not comparisons), if appropriate, list up to 3 laptops in total (1 primary, up to 2 alternatives).
    - If, after reviewing the `{context}`, no laptop seems to be a good match for the understood query, clearly state that you couldn't find a suitable option with the available information.
    - Base all information on the provided `{context}` or `Chat History`. Do not invent details.
    - Keep responses user-friendly, clear, and concise.

    ---
    Now, please generate your helpful answer.

    Context:
    {context}

    Chat History:
    {history}

    User Query:
    {question}

    Helpful Answer:
    """
    
    prompt = PromptTemplate(input_variables=["context", "history", "question"], template=template)
    
    # metadata_filter function remains the same as in your provided code.
    # It plays a crucial role in providing a relevant {context} to the LLM,
    # effectively contributing to Step 2 (Parse Query - by extracting hard filters) and Step 3 (RAG Retrieve - by filtering documents).
    def metadata_filter(input_dict):
        query = input_dict["query"].lower()
        docs = vectorstore.similarity_search(query, k=10) 
        filtered_docs = []
        
        price_min, price_max = None, None
        price_patterns = [
            r'(\d+\.?\d*k?)\s*(?:rs|rupees)?\s*(?:to|-)\s*(\d+\.?\d*k?)\s*(?:rs|rupees)?',
            r'between\s*(\d+\.?\d*k?)\s*(?:rs|rupees)?\s*and\s*(\d+\.?\d*k?)\s*(?:rs|rupees)?',
            r'(?:under|below|max)\s*(\d+\.?\d*k?)\s*(?:rs|rupees)?',
            r'(?:above|over|min)\s*(\d+\.?\d*k?)\s*(?:rs|rupees)?',
        ]

        def parse_price_value(value_str):
            value_str = str(value_str).lower().replace(",", "") # Ensure string, remove commas
            multiplier = 1000 if 'k' in value_str else 1
            # Remove non-numeric characters except decimal point before float conversion
            cleaned_value_str = re.sub(r'[^\d.]', '', value_str.replace('k','')) 
            if not cleaned_value_str: return None
            return float(cleaned_value_str) * multiplier
        
        for pattern in price_patterns:
            match = re.search(pattern, query)
            if match:
                groups = [g for g in match.groups() if g is not None] 
                if pattern.startswith(r'(\d+\.?\d*k?)\s*(?:rs|rupees)?\s*(?:to|-)'): 
                    if len(groups) == 2:
                        price_min = parse_price_value(groups[0])
                        price_max = parse_price_value(groups[1])
                elif pattern.startswith(r'between\s*(\d+\.?\d*k?)'): 
                     if len(groups) == 2:
                        price_min = parse_price_value(groups[0])
                        price_max = parse_price_value(groups[1])
                elif pattern.startswith(r'(?:under|below|max)'): 
                    if len(groups) == 1:
                        price_max = parse_price_value(groups[0])
                elif pattern.startswith(r'(?:above|over|min)'): 
                    if len(groups) == 1:
                        price_min = parse_price_value(groups[0])
                if price_min is not None or price_max is not None:
                    break
        
        ram_min = None
        ram_match = re.search(r'(\d+)\s*(?:gb)?\s*ram', query)
        if ram_match:
            ram_min = int(ram_match.group(1))
        
        storage_min = None
        storage_type_query = None
        storage_match = re.search(r'(\d+)\s*(gb|tb)\s*(ssd|hdd)?', query)
        if storage_match:
            size, unit, type_ = storage_match.groups()
            storage_min = int(size) * (1024 if unit.lower() == "tb" else 1)
            if type_:
                storage_type_query = type_.upper()
        
        processor_query_pattern = None
        processor_patterns = [
            r'intel\s*core\s*(i[3579]\s*\d{2,5}\w?|\d{2,5}\w?\s*i[3579]|ultra\s*[579])',
            r'amd\s*ryzen\s*([3579]\s*\d{4}\w?)' 
        ]
        for p_pattern in processor_patterns:
            match = re.search(p_pattern, query)
            if match:
                processor_query_pattern = match.group(0) 
                break
        if not processor_query_pattern: 
            match = re.search(r'(intel\s*core\s*[^\s,]+|amd\s*ryzen\s*[^\s,]+)', query)
            if match:
                processor_query_pattern = match.group(0)

        display_size_query = None
        display_match = re.search(r'(\d+\.?\d*)\s*inch', query)
        if display_match:
            display_size_query = display_match.group(1)

        for doc in docs:
            metadata = doc.metadata
            if price_min and (not metadata["price"] or metadata["price"] < price_min):
                continue
            if price_max and (not metadata["price"] or metadata["price"] > price_max):
                continue
            if ram_min and (not metadata["ram"] or metadata["ram"] < ram_min):
                continue
            if storage_min:
                if not metadata["storage"]: continue
                doc_storage_match = re.search(r'(\d+)', str(metadata["storage"]))
                if not doc_storage_match or int(doc_storage_match.group(1)) < storage_min:
                    continue
            if storage_type_query:
                if not metadata["storage"] or storage_type_query not in str(metadata["storage"]).upper():
                    continue
            if processor_query_pattern:
                if not metadata["processor"] or not re.search(processor_query_pattern, metadata["processor"].lower()):
                    continue
            if display_size_query:
                if not metadata["display_size"]: continue
                doc_display_match = re.search(r'\((\d+\.?\d*)\s*inch\)', str(metadata["display_size"]))
                if not doc_display_match or doc_display_match.group(1) != display_size_query:
                    continue
            filtered_docs.append(doc)
        
        return filtered_docs[:5]

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)
    
    retriever = RunnableLambda(metadata_filter)
    format_docs_runnable = RunnableLambda(format_docs)
    
    rag_chain = RunnableSequence(
        {
            "context": retriever | format_docs_runnable,
            "history": RunnableLambda(lambda x: x.get("history", "")),
            "question": RunnableLambda(lambda x: x["query"])
        },
        prompt,
        llm
    )
    
    return rag_chain

# Query Processing (refine_query, process_query - structure remains, history formatting updated)
# Global variable for RAG chain, to be initialized in main()
rag_chain = None 

def refine_query(query):
    """Refine user query for better search."""
    query = query.strip().lower()
    query = re.sub(r'\bi([3579])\b', r'intel core i\1', query) 
    query = re.sub(r'\bryzen\s*([3579])\b', r'amd ryzen \1', query) 
    return query

def process_query(query, history_tuples_list):
    """Process user query with chat history."""
    if not query:
        return "Please enter a valid query."
    
    if rag_chain is None:
        return "Error: The recommendation system is not initialized. Please restart."

    refined_query = refine_query(query)
    
    formatted_history = []
    if history_tuples_list: 
        for user_msg, bot_msg in history_tuples_list[-5:]: 
            formatted_history.append(f"User: {user_msg}\nAssistant: {bot_msg}")
    history_str = "\n\n".join(formatted_history)
    
    try:
        result = rag_chain.invoke({"query": refined_query, "history": history_str})
        return result.content
    except Exception as e:
        print(f"Error during RAG chain invocation: {str(e)}") 
        return f"Sorry, I encountered an error: {str(e)}. Please try again."

# Gradio App (main function - structure remains, minor Gradio UI tweaks possible)
def main():
    """Create and launch Gradio app."""
    global rag_chain
    CSV_FILE = "flipkart_laptop_cleaned.csv" 
    try:
        chunks = process_data(CSV_FILE)
        rag_chain = setup_rag(chunks) 
    except Exception as e:
        print(f"FATAL: Error initializing RAG chain: {str(e)}")
        # This error needs to be handled gracefully if Gradio is to show a message
        # For now, it will prevent the app from starting correctly.
        # A more robust solution would involve Gradio displaying this startup error.
        # For simplicity, we raise it, which stops before Gradio launch if init fails.
        raise ValueError(f"Error initializing app backend: {str(e)}")
    
    def chat_interface_fn(query, history):
        if rag_chain is None:
            return "The recommendation system isn't ready. Please ensure it started correctly."
        return process_query(query, history) # history from Gradio is list of (user, bot) tuples
    
    app = gr.ChatInterface(
        fn=chat_interface_fn,
        title="Laptop Recommendation System",
        description="Ask for laptop recommendations based on price, specs, or use case (e.g., 'laptops between 50,000 Rs to 80,000 Rs with SSD' or 'laptop for a software engineer'). You can also ask to compare recommended laptops.",
        theme="soft",
        examples=[
            ["budget laptop for remote work"],
            ["laptops for students under 40000 rs with 8gb ram"],
            ["gaming laptop with rtx 4060 around 1 lakh"],
            ["compare the first two laptops you mentioned"],
            ["ultrabook for travel, good battery, under 70k, lightweight"]
        ],
        chatbot=gr.Chatbot(height=600, label="Chat Window"),
        textbox=gr.Textbox(placeholder="Tell me what you're looking for...", container=False, scale=7, label="Your Query"),
        retry_btn="Retry",
        undo_btn="Delete previous turn",
        clear_btn="Clear chat",
    )
    
    app.launch(share=False, debug=True)

if __name__ == "__main__":
    main()
