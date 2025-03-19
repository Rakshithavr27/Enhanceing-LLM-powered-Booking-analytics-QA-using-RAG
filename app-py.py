import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, List, Any, Optional
import uvicorn
import os
import time
from datetime import datetime
import json
import re
import torch
from sentence_transformers import SentenceTransformer
import faiss
from transformers import AutoTokenizer, AutoModelForCausalLM

app = FastAPI(title="Hotel Booking Analytics & QA API")

# Global data storage
DATA_FILE = "hotel_bookings.csv"
QUERY_HISTORY_FILE = "query_history.json"
loaded_data = None
analytics_engine = None
rag_system = None

class QueryRequest(BaseModel):
    question: str

class AnalyticsRequest(BaseModel):
    time_period: Optional[str] = "monthly"
    specific_analysis: Optional[List[str]] = None

def load_and_preprocess_data(file_path):
    """Load and preprocess the hotel booking data"""
    # Load data
    df = pd.read_csv(file_path)
    
    # Handle missing values
    df.fillna({
        'children': 0,
        'country': 'unknown',
        'agent': 0,
        'company': 0
    }, inplace=True)
    
    # Convert dates to datetime objects
    df['reservation_status_date'] = pd.to_datetime(df['reservation_status_date'])
    df['arrival_date'] = pd.to_datetime(df['arrival_date_year'].astype(str) + '-' + 
                                      df['arrival_date_month'] + '-' + 
                                      df['arrival_date_day_of_month'].astype(str))
    
    # Calculate total price
    df['total_price'] = df['adr'] * df['stays_in_weekend_nights'] + df['adr'] * df['stays_in_week_nights']
    
    # Calculate booking lead time in days
    df['lead_time_days'] = df['lead_time']
    
    # Create a is_canceled binary column (0 or 1)
    df['is_canceled_binary'] = df['is_canceled']
    
    return df

class BookingAnalytics:
    def __init__(self, data):
        self.data = data
        
    def revenue_trends(self, time_period='monthly'):
        """Calculate revenue trends over time."""
        if time_period == 'monthly':
            self.data['month_year'] = self.data['arrival_date'].dt.to_period('M')
            revenue_by_time = self.data.groupby('month_year')['total_price'].sum().reset_index()
            revenue_by_time['month_year'] = revenue_by_time['month_year'].astype(str)
            return revenue_by_time
        elif time_period == 'quarterly':
            self.data['quarter_year'] = self.data['arrival_date'].dt.to_period('Q')
            revenue_by_time = self.data.groupby('quarter_year')['total_price'].sum().reset_index()
            revenue_by_time['quarter_year'] = revenue_by_time['quarter_year'].astype(str)
            return revenue_by_time
        else:
            self.data['year'] = self.data['arrival_date'].dt.year
            revenue_by_time = self.data.groupby('year')['total_price'].sum().reset_index()
            return revenue_by_time
    
    def cancellation_rate(self):
        """Calculate cancellation rate as percentage of total bookings."""
        total_bookings = len(self.data)
        cancelled_bookings = self.data['is_canceled'].sum()
        cancellation_rate = (cancelled_bookings / total_bookings) * 100
        
        cancellation_by_hotel = self.data.groupby('hotel')['is_canceled'].mean() * 100
        
        return {
            'overall_cancellation_rate': cancellation_rate,
            'cancellation_by_hotel': cancellation_by_hotel.to_dict()
        }
    
    def geographical_distribution(self):
        """Get geographical distribution of users."""
        geo_distribution = self.data['country'].value_counts().reset_index()
        geo_distribution.columns = ['country', 'count']
        
        return geo_distribution.to_dict(orient='records')
    
    def lead_time_distribution(self):
        """Analyze booking lead time distribution."""
        # Create lead time bins
        bins = [0, 7, 30, 90, 180, 365, float('inf')]
        labels = ['Less than a week', '1-4 weeks', '1-3 months', '3-6 months', '6-12 months', 'More than a year']
        
        self.data['lead_time_category'] = pd.cut(self.data['lead_time'], bins=bins, labels=labels)
        lead_time_dist = self.data['lead_time_category'].value_counts().reset_index()
        lead_time_dist.columns = ['category', 'count']
        
        # Calculate statistics
        lead_time_stats = {
            'mean_lead_time': self.data['lead_time'].mean(),
            'median_lead_time': self.data['lead_time'].median(),
            'distribution': lead_time_dist.to_dict(orient='records')
        }
        
        return lead_time_stats
    
    def additional_analytics(self):
        """Generate additional insights."""
        # Average daily rate by hotel type and month
        adr_by_hotel_month = self.data.groupby(['hotel', 'arrival_date_month'])['adr'].mean().reset_index()
        
        # Guest composition
        guest_composition = {
            'with_children': (self.data['children'] > 0).mean() * 100,
            'with_babies': (self.data['babies'] > 0).mean() * 100,
            'average_guests_per_booking': self.data['adults'].mean() + self.data['children'].mean() + self.data['babies'].mean()
        }
        
        # Market segment analysis
        market_segment = self.data['market_segment'].value_counts(normalize=True) * 100
        
        # Repeat guests analysis
        repeat_guests = {
            'percentage': (self.data['is_repeated_guest'].mean() * 100),
            'cancellation_rate': self.data[self.data['is_repeated_guest'] == 1]['is_canceled'].mean() * 100,
            'average_adr': self.data[self.data['is_repeated_guest'] == 1]['adr'].mean()
        }
        
        return {
            'adr_by_hotel_month': adr_by_hotel_month.to_dict(orient='records'),
            'guest_composition': guest_composition,
            'market_segment': market_segment.to_dict(),
            'repeat_guests': repeat_guests
        }
    
    def generate_all_analytics(self):
        """Generate all analytics in one call."""
        return {
            'revenue_trends': self.revenue_trends().to_dict(orient='records'),
            'cancellation_rate': self.cancellation_rate(),
            'geographical_distribution': self.geographical_distribution(),
            'lead_time_distribution': self.lead_time_distribution(),
            'additional_analytics': self.additional_analytics()
        }

class RAGSystem:
    def __init__(self, data_df, model_name="mistralai/Mistral-7B-Instruct-v0.2"):
        self.data = data_df
        # Convert our dataframe to documents
        self.documents = self._prepare_documents()
        
        # Initialize embedding model
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Initialize LLM for answering - Comment this if not using GPU
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name, 
                torch_dtype=torch.float16, 
                device_map="auto",
                load_in_8bit=True  # Use 8-bit quantization for memory efficiency
            )
            self.model_loaded = True
        except Exception as e:
            print(f"Warning: Could not load LLM model: {e}")
            self.model_loaded = False
    
    def _prepare_documents(self):
        """Convert dataframe rows to document strings for embedding"""
        documents = []
        
        # First, prepare documents for each month's aggregated data
        monthly_data = self.data.groupby(pd.Grouper(key='arrival_date', freq='M')).agg({
            'total_price': 'sum',
            'is_canceled': 'mean',
            'adr': 'mean',
            'hotel': 'count'
        }).reset_index()
        
        for _, row in monthly_data.iterrows():
            month_year = row['arrival_date'].strftime('%B %Y')
            doc = (f"For {month_year}, the total revenue was ${row['total_price']:.2f}, "
                   f"with {row['hotel']} bookings. "
                   f"The cancellation rate was {row['is_canceled']*100:.2f}% and "
                   f"the average daily rate was ${row['adr']:.2f}.")
            documents.append(doc)
        
        # Add country-specific documents
        for country, group in self.data.groupby('country'):
            if len(group) > 10:  # Only consider countries with meaningful data
                doc = (f"For guests from {country}, there were {len(group)} bookings "
                       f"with a cancellation rate of {group['is_canceled'].mean()*100:.2f}%. "
                       f"The average price was ${group['adr'].mean():.2f} per night.")
                documents.append(doc)
        
        # Add hotel-specific documents
        for hotel, group in self.data.groupby('hotel'):
            doc = (f"The {hotel} had {len(group)} bookings with an average price of "
                   f"${group['adr'].mean():.2f} per night and a cancellation rate of "
                   f"{group['is_canceled'].mean()*100:.2f}%.")
            documents.append(doc)
        
        # Add some specific statistics documents
        docs = [
            f"The overall average price of a hotel booking is ${self.data['adr'].mean():.2f} per night.",
            f"The overall cancellation rate is {self.data['is_canceled'].mean()*100:.2f}% of total bookings.",
            f"The average lead time for bookings is {self.data['lead_time'].mean():.1f} days.",
            f"The most common booking market segment is {self.data['market_segment'].value_counts().index[0]}."
        ]
        documents.extend(docs)
        
        return documents
    
    def _create_index(self):
        """Create FAISS index for fast retrieval"""
        # Generate embeddings for all documents
        self.embeddings = self.embedding_model.encode(self.documents)
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(self.embeddings)
        
        # Build the index
        embedding_dim = self.embeddings.shape[1]
        self.index = faiss.IndexFlatIP(embedding_dim)  # Inner product for cosine similarity
        self.index.add(self.embeddings)
    
    def retrieve(self, query, top_k=5):
        """Retrieve relevant documents for a query"""
        # Generate embedding for the query
        query_embedding = self.embedding_model.encode([query])
        faiss.normalize_L2(query_embedding)
        
        # Search for similar documents
        scores, indices = self.index.search(query_embedding, top_k)
        
        # Return relevant documents
        relevant_docs = [self.documents[idx] for idx in indices[0]]
        return relevant_docs
    
    def answer_question(self, question):
        """Answer a question using RAG"""
        # Retrieve relevant context
        context = self.retrieve(question)
        context_text = "\n".join(context)
        
        if not self.model_loaded:
            # Fallback if model isn't loaded
            return {
                "question": question,
                "answer": f"Based on the retrieved information: {context_text}",
                "context": context
            }
        
        # Prepare prompt for the LLM
        prompt = f"""<s>[INST] You are a helpful assistant for a hotel booking system. 
Answer the question based only on the context provided. If you can't find the answer in the context, say "I don't have enough information to answer that."

Context:
{context_text}

Question: {question} [/INST]"""
        
        # Generate answer with the LLM
        try:
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
            outputs = self.model.generate(
                inputs["input_ids"],
                max_new_tokens=256,
                temperature=0.7,
                top_p=0.9,
            )
            answer = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract only the assistant's response
            answer = answer.split("[/INST]")[1].strip() if "[/INST]" in answer else answer
        except Exception as e:
            answer = f"Error generating response: {str(e)}. Based on the retrieved information: {context_text}"
        
        return {
            "question": question,
            "answer": answer,
            "context": context
        }

    def generate_specific_answer(self, query):
        """Generate more accurate answers for specific query types"""
        # Try to identify query patterns for direct data lookup
        
        # Pattern for revenue in a specific month and year
        revenue_pattern = r"revenue\s+for\s+(\w+)\s+(\d{4})"
        match = re.search(revenue_pattern, query, re.IGNORECASE)
        
        if match:
            month, year = match.groups()
            # Direct calculation from data
            try:
                filtered_data = self.data[
                    (self.data['arrival_date'].dt.month == pd.to_datetime(month, format='%B').month) & 
                    (self.data['arrival_date'].dt.year == int(year))
                ]
                if not filtered_data.empty:
                    total_revenue = filtered_data['total_price'].sum()
                    return {
                        "question": query,
                        "answer": f"The total revenue for {month} {year} was ${total_revenue:.2f}.",
                        "context": [f"Direct calculation: {len(filtered_data)} bookings in {month} {year} with total revenue ${total_revenue:.2f}"]
                    }
            except Exception:
                pass  # Fall back to RAG answer if direct lookup fails
        
        # If no specific pattern matched, fall back to RAG answer
        return self.answer_question(query)

def load_or_create_query_history():
    if os.path.exists(QUERY_HISTORY_FILE):
        with open(QUERY_HISTORY_FILE, 'r') as f:
            return json.load(f)
    else:
        return []

# Initialize system on startup
@app.on_event("startup")
async def startup_db_client():
    global loaded_data, analytics_engine, rag_system
    
    try:
        # Load and preprocess data
        loaded_data = load_and_preprocess_data(DATA_FILE)
        
        # Initialize analytics engine
        analytics_engine = BookingAnalytics(loaded_data)
        
        # Initialize RAG system
        rag_system = RAGSystem(loaded_data)
        rag_system._create_index()  # Create index at startup
        
        print("System initialized successfully!")
    except Exception as e:
        print(f"Error initializing system: {e}")

@app.get("/")
async def root():
    return {"message": "Welcome to Hotel Booking Analytics & QA API"}

@app.post("/analytics")
async def get_analytics(request: AnalyticsRequest):
    if not analytics_engine:
        raise HTTPException(status_code=503, detail="Analytics engine not initialized")
    
    start_time = time.time()
    
    try:
        # If specific analyses are requested
        if request.specific_analysis:
            results = {}
            for analysis in request.specific_analysis:
                if analysis == "revenue_trends":
                    results[analysis] = analytics_engine.revenue_trends(request.time_period).to_dict(orient='records')
                elif analysis == "cancellation_rate":
                    results[analysis] = analytics_engine.cancellation_rate()
                elif analysis == "geographical_distribution":
                    results[analysis] = analytics_engine.geographical_distribution()
                elif analysis == "lead_time_distribution":
                    results[analysis] = analytics_engine.lead_time_distribution()
                elif analysis == "additional_analytics":
                    results[analysis] = analytics_engine.additional_analytics()
        else:
            # Return all analytics
            results = analytics_engine.generate_all_analytics()
        
        processing_time = time.time() - start_time
        
        return {
            "data": results,
            "processing_time_seconds": processing_time
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating analytics: {str(e)}")

@app.post("/ask")
async def ask_question(request: QueryRequest):
    if not rag_system:
        raise HTTPException(status_code=503, detail="RAG system not initialized")
    
    start_time = time.time()
    
    try:
        # Generate answer
        result = rag_system.generate_specific_answer(request.question)
        
        # Calculate processing time
        processing_time = time.time() - start_time
        
        # Store query in history
        query_history = load_or_create_query_history()
        query_history.append({
            "timestamp": datetime.now().isoformat(),
            "question": request.question,
            "processing_time": processing_time
        })
        
        with open(QUERY_HISTORY_FILE, 'w') as f:
            json.dump(query_history, f)
        
        # Return results with processing time
        return {
            "question": request.question,
            "answer": result["answer"],
            "processing_time_seconds": processing_time
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing question: {str(e)}")

@app.get("/health")
async def check_health():
    """Check the health status of the system"""
    health_status = {
        "status": "healthy",
        "components": {
            "database": loaded_data is not None,
            "analytics_engine": analytics_engine is not None,
            "rag_system": rag_system is not None,
            "rag_index": hasattr(rag_system, 'index'),
            "llm_model": getattr(rag_system, 'model_loaded', False)
        },
        "timestamp": datetime.now().isoformat()
    }
    
    # Overall status
    if not all(health_status["components"].values()):
        health_status["status"] = "unhealthy"
        
    return health_status

@app.get("/query_history")
async def get_query_history():
    """Get the history of queries asked to the system"""
    try:
        query_history = load_or_create_query_history()
        return {"history": query_history}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving query history: {str(e)}")

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
