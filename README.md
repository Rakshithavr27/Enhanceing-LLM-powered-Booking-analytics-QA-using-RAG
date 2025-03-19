# LLM-Powered Hotel Booking Analytics & QA System

This project implements a comprehensive system for analyzing hotel booking data and providing question-answering capabilities through a REST API. The system processes booking data, extracts insights, and enables retrieval-augmented question answering (RAG).

## Features

- **Data Processing**: Handles hotel booking data with cleaning and preprocessing
- **Analytics**: Generates insights on revenue trends, cancellation rates, geographical distribution, and booking lead times
- **RAG Question Answering**: Uses vector embeddings and LLM to answer natural language questions about the data
- **REST API**: Provides endpoints for analytics and question answering
- **Performance Evaluation**: Tools to measure system accuracy and response time
- **Health Monitoring**: Endpoint to check system status
- **Query History**: Tracks questions asked to the system

## Getting Started

### Prerequisites

- Python 3.8+
- pip package manager

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/hotel-booking-analytics.git
   cd hotel-booking-analytics
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   # On Windows
   venv\Scripts\activate
   # On Unix/MacOS
   source venv/bin/activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements-file.txt
   ```

4. Download the dataset:
   - Download the hotel booking dataset from [Kaggle](https://www.kaggle.com/jessemostipak/hotel-booking-demand)
   - Save it as `hotel_bookings.csv` in the project root directory

### Running the System

1. Start the API server:
   ```bash
   python app.py
   ```

2. Access the API at `http://localhost:8000`

3. API Documentation is available at `http://localhost:8000/docs`

## API Endpoints

### POST /analytics
Returns analytics reports based on the hotel booking data.

**Request Body**:
```json
{
  "time_period": "monthly",
  "specific_analysis": ["revenue_trends", "cancellation_rate"]
}
```

**Response**:
```json
{
  "data": {
    "revenue_trends": [...],
    "cancellation_rate": {...}
  },
  "processing_time_seconds": 0.123
}
```

### POST /ask
Answers questions about the booking data using RAG.

**Request Body**:
```json
{
  "question": "Show me total revenue for July 2017"
}
```

**Response**:
```json
{
  "question": "Show me total revenue for July 2017",
  "answer": "The total revenue for July 2017 was $1,234,567.89.",
  "processing_time_seconds": 0.456
}
```

### GET /health
Checks the system status.

**Response**:
```json
{
  "status": "healthy",
  "components": {
    "database": true,
    "analytics_engine": true,
    "rag_system": true
  },
  "timestamp": "2023-09-15T12:34:56.789Z"
}
```

### GET /query_history
Returns the history of queries asked to the system.

**Response**:
```json
{
  "history": [
    {
      "timestamp": "2023-09-15T12:34:56.789Z",
      "question": "Show me total revenue for July 2017",
      "processing_time": 0.456
    },
    ...
  ]
}
```

## System Architecture

The system consists of the following components:

1. **Data Preprocessing**: Cleans and prepares the hotel booking data
2. **Analytics Engine**: Generates insights and visualizations
3. **RAG System**: Combines FAISS vector database with LLM for question answering
4. **API Layer**: Provides REST endpoints using FastAPI

## Implementation Choices & Challenges

### LLM Selection
We chose Mistral-7B-Instruct for its balance of performance and resource requirements. Alternative options included Llama-2 and Falcon, but Mistral provided better responses for our specific domain.

### Vector Database
FAISS was selected for its efficiency in similarity search operations and ease of integration. It allows us to quickly retrieve relevant booking data contexts when answering questions.

### API Design
FastAPI was chosen for its performance, automatic documentation generation, and type checking capabilities. The async support allows for better handling of concurrent requests.

### Challenges

1. **Data Preprocessing**: The hotel booking dataset required significant cleaning, especially for date fields and handling missing values.

2. **RAG System Optimization**: Balancing between retrieval accuracy and processing speed was challenging. We improved performance by:
   - Pre-computing embeddings for common queries
   - Creating specialized document chunks for different query types
   - Implementing direct calculation for specific query patterns

3. **Response Quality**: Ensuring that the LLM provided accurate, relevant answers required careful prompt engineering and context selection.

## Sample Test Queries

Here are some example queries and their expected answers:

1. "Show me total revenue for July 2017."
   - Expected: A precise figure of the total revenue for that specific month

2. "Which locations had the highest booking cancellations?"
   - Expected: A list of countries or hotels with the highest cancellation rates

3. "What is the average price of a hotel booking?"
   - Expected: The overall average daily rate across all bookings

4. "What's the trend in booking lead times over the year?"
   - Expected: Analysis of how far in advance bookings are made throughout the year

5. "Compare cancellation rates between resort and city hotels."
   - Expected: Statistical comparison between the two hotel types

## Future Improvements

- Implement real-time data updates using a database like PostgreSQL
- Add user authentication for API access
- Enhance the RAG system with multi-hop reasoning capabilities
- Develop a web frontend for visualization and interaction
- Add support for more languages in the question answering system

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgements

- Dataset: Hotel Booking Demand dataset by Jesse Mostipak on Kaggle
- Libraries: Pandas, NumPy, FAISS, Transformers, FastAPI
