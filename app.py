import json
import sqlite3
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List, Tuple
import re
import logging
from datetime import datetime, timedelta
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import openai
import pickle
import os
from collections import defaultdict
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import traceback

# Load environment variables from .env if available
try:
    from dotenv import load_dotenv  # type: ignore
    load_dotenv()
except Exception:
    # Minimal manual loader as a fallback
    env_path = os.path.join(os.path.dirname(__file__), '.env')
    if os.path.exists(env_path):
        try:
            with open(env_path, 'r') as _f:
                for _line in _f:
                    _line = _line.strip()
                    if not _line or _line.startswith('#') or '=' not in _line:
                        continue
                    _k, _v = _line.split('=', 1)
                    os.environ.setdefault(_k.strip(), _v.strip())
        except Exception:
            pass

logger = logging.getLogger(__name__)

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

def configure_openai() -> bool:
    """Configure the OpenAI client with the API key.

    Returns True when the key is set; False otherwise.
    """
    try:
        if not OPENAI_API_KEY:
            logger.warning("OpenAI API key not set. Create a .env with OPENAI_API_KEY or set it in environment.")
            return False
        openai.api_key = OPENAI_API_KEY
        return True
    except Exception as e:
        logger.error(f"Failed to configure OpenAI: {e}")
        return False

class ComprehensiveFinancialRAG:
    def __init__(self, db_path: str = "financials.db"):
        self.db_path = db_path
        self.training_data_path = "comprehensive_training.json"
        self.vectorizer_path = "financial_vectorizer.pkl"
        self.embeddings_path = "financial_embeddings.pkl"
        self.stats_path = "query_stats.json"
        
        # Core components
        self.schema = self.get_schema()
        self.training_examples = self.load_comprehensive_training_data()
        self.vectorizer = None
        self.example_embeddings = None
        self.query_stats = self.load_query_stats()
        
        # Initialize RAG system
        self.initialize_comprehensive_rag()
        logger.info(f"Initialized with {len(self.training_examples)} training examples")

    def load_comprehensive_training_data(self) -> List[Dict]:
        """Load comprehensive training data from the provided examples"""
        try:
            if os.path.exists(self.training_data_path):
                with open(self.training_data_path, 'r') as f:
                    return json.load(f)
            else:
                return self.create_comprehensive_training_set()
        except Exception as e:
            logger.error(f"Error loading training data: {e}")
            return self.create_comprehensive_training_set()

    def create_comprehensive_training_set(self) -> List[Dict]:
        """Create comprehensive training set from all provided examples"""
        training_set = []
        
        # Simple queries - Basic operations
        simple_queries = [
            {
                "id": "simple_1",
                "question": "Fetch Apple's most recent closing price",
                "sql": "SELECT symbol, date, close FROM historical_prices WHERE symbol = 'AAPL' ORDER BY date DESC LIMIT 1",
                "category": "stock_price",
                "complexity": "simple",
                "intent": "latest_price",
                "success_count": 10,
                "explanation": "Single table query with specific company filter and ordering"
            },
            {
                "id": "simple_2",
                "question": "Show companies classified under the Technology sector",
                "sql": "SELECT symbol, name, sector FROM sp500_constituents WHERE sector = 'Technology' ORDER BY name",
                "category": "company_info",
                "complexity": "simple", 
                "intent": "sector_filter",
                "success_count": 8,
                "explanation": "Filter companies by sector"
            },
            {
                "id": "simple_3",
                "question": "Return Walmart's revenue from the latest reported quarter",
                "sql": "SELECT symbol, revenue, date FROM income_statements_quarterly WHERE symbol = 'WMT' ORDER BY date DESC LIMIT 1",
                "category": "financial_metrics",
                "complexity": "simple",
                "intent": "latest_financial",
                "success_count": 12,
                "explanation": "Get latest financial metric for specific company"
            },
            {
                "id": "simple_4",
                "question": "Provide Microsoft's most recent total assets figure",
                "sql": "SELECT symbol, totalAssets, date FROM balance_sheets_quarterly WHERE symbol = 'MSFT' ORDER BY date DESC LIMIT 1",
                "category": "balance_sheet",
                "complexity": "simple",
                "intent": "balance_sheet_item",
                "success_count": 9,
                "explanation": "Get latest balance sheet item for specific company"
            },
            {
                "id": "simple_5",
                "question": "Show Amazon's latest free cash flow value",
                "sql": "SELECT symbol, freeCashFlow, date FROM cash_flow_statements_quarterly WHERE symbol = 'AMZN' ORDER BY date DESC LIMIT 1",
                "category": "cash_flow",
                "complexity": "simple",
                "intent": "cash_flow_metric",
                "success_count": 11,
                "explanation": "Get latest cash flow metric for specific company"
            },
            {
                "id": "simple_6",
                "question": "List companies in the Healthcare sector",
                "sql": "SELECT symbol, name, sector, headQuarter FROM sp500_constituents WHERE sector = 'Healthcare' ORDER BY name",
                "category": "company_info",
                "complexity": "simple",
                "intent": "sector_filter",
                "success_count": 7,
                "explanation": "Filter companies by healthcare sector"
            },
            {
                "id": "annual_1",
                "question": "Report Apple's most recent annual revenue",
                "sql": "SELECT symbol, revenue, date, period FROM income_statements_annual WHERE symbol = 'AAPL' ORDER BY date DESC LIMIT 1",
                "category": "annual_financials",
                "complexity": "simple",
                "intent": "annual_revenue",
                "success_count": 15,
                "explanation": "Get latest annual revenue for specific company"
            },
            {
                "id": "annual_4",
                "question": "Return Amazon's total assets from the latest fiscal year",
                "sql": "SELECT symbol, totalAssets, totalLiabilities, date FROM balance_sheets_annual WHERE symbol = 'AMZN' ORDER BY date DESC LIMIT 1",
                "category": "annual_balance_sheet",
                "complexity": "simple",
                "intent": "annual_balance",
                "success_count": 13,
                "explanation": "Get annual balance sheet data"
            },
            {
                "id": "annual_6",
                "question": "Show Google's most recent annual operating cash flow",
                "sql": "SELECT symbol, operatingCashFlow, freeCashFlow, date FROM cash_flow_statements_annual WHERE symbol = 'GOOGL' ORDER BY date DESC LIMIT 1",
                "category": "annual_cash_flow",
                "complexity": "simple",
                "intent": "annual_cash_flow",
                "success_count": 14,
                "explanation": "Get annual cash flow metrics"
            },
            # Q2 2025 specific examples
            {
                "id": "q2_2025_1",
                "question": "Which company recorded the highest revenue in Q2 2025?",
                "sql": "SELECT sc.name, isq.symbol, isq.revenue, isq.date FROM income_statements_quarterly isq JOIN sp500_constituents sc ON isq.symbol = sc.symbol WHERE isq.date >= '2025-04-01' AND isq.date <= '2025-06-30' AND isq.revenue IS NOT NULL ORDER BY isq.revenue DESC LIMIT 1",
                "category": "quarterly_revenue",
                "complexity": "simple",
                "intent": "quarterly_ranking",
                "success_count": 20,
                "explanation": "Q2 means April-June, use date range filtering with single SELECT"
            },
            {
                "id": "q2_2025_2",
                "question": "Name the revenue leader for Q2 2025",
                "sql": "SELECT sc.name, isq.symbol, isq.revenue, isq.date FROM income_statements_quarterly isq JOIN sp500_constituents sc ON isq.symbol = sc.symbol WHERE isq.date >= '2025-04-01' AND isq.date <= '2025-06-30' AND isq.revenue IS NOT NULL ORDER BY isq.revenue DESC LIMIT 1",
                "category": "quarterly_revenue",
                "complexity": "simple",
                "intent": "quarterly_ranking",
                "success_count": 18,
                "explanation": "Single SELECT statement for Q2 revenue query - avoid UNION or combining unrelated data"
            }
        ]
        
        # Medium complexity queries
        medium_queries = [
            {
                "id": "medium_1",
                "question": "Top 5 companies by revenue in the latest quarter",
                "sql": "SELECT sc.symbol, sc.name, isq.revenue, isq.date FROM income_statements_quarterly isq JOIN sp500_constituents sc ON isq.symbol = sc.symbol WHERE isq.date >= '2025-01-01' AND isq.revenue IS NOT NULL ORDER BY isq.revenue DESC LIMIT 5",
                "category": "ranking",
                "complexity": "medium",
                "intent": "top_companies",
                "success_count": 25,
                "explanation": "Rank companies by revenue with JOIN and date filtering"
            },
            {
                "id": "medium_2",
                "question": "Compare Apple vs Microsoft closing prices for the past month",
                "sql": "SELECT symbol, date, close FROM historical_prices WHERE symbol IN ('AAPL', 'MSFT') AND date >= '2025-05-24' ORDER BY symbol, date DESC",
                "category": "comparison",
                "complexity": "medium",
                "intent": "price_comparison",
                "success_count": 22,
                "explanation": "Compare multiple companies over time period using IN clause"
            },
            {
                "id": "medium_3",
                "question": "Among Technology sector firms, which have the highest profit margins?",
                "sql": "SELECT sc.symbol, sc.name, ROUND((CAST(isq.netIncome AS REAL) / CAST(isq.revenue AS REAL)) * 100, 2) AS profit_margin FROM income_statements_quarterly isq JOIN sp500_constituents sc ON isq.symbol = sc.symbol WHERE sc.sector = 'Technology' AND isq.date >= '2025-01-01' AND isq.revenue > 0 AND isq.netIncome IS NOT NULL ORDER BY profit_margin DESC LIMIT 10",
                "category": "calculation",
                "complexity": "medium",
                "intent": "profit_margin",
                "success_count": 30,
                "explanation": "Calculate profit margins with sector filtering and mathematical operations"
            },
            {
                "id": "medium_4",
                "question": "Companies generating over $100B in quarterly revenue",
                "sql": "SELECT sc.symbol, sc.name, isq.revenue FROM income_statements_quarterly isq JOIN sp500_constituents sc ON isq.symbol = sc.symbol WHERE isq.revenue > 100000000000 AND isq.date >= '2025-01-01' ORDER BY isq.revenue DESC",
                "category": "filtering",
                "complexity": "medium",
                "intent": "revenue_threshold",
                "success_count": 19,
                "explanation": "Filter by revenue threshold with JOIN"
            },
            {
                "id": "medium_5",
                "question": "Average closing price by sector (recent period)",
                "sql": "SELECT sc.sector, ROUND(AVG(hp.close), 2) AS avg_price FROM historical_prices hp JOIN sp500_constituents sc ON hp.symbol = sc.symbol WHERE hp.date >= '2025-06-01' GROUP BY sc.sector ORDER BY avg_price DESC",
                "category": "aggregation",
                "complexity": "medium",
                "intent": "sector_analysis",
                "success_count": 35,
                "explanation": "Calculate sector averages using GROUP BY with JOIN"
            },
            {
                "id": "medium_6",
                "question": "Which companies have the highest debt-to-equity ratios?",
                "sql": "SELECT sc.symbol, sc.name, bs.longTermDebt, bs.totalStockholdersEquity, ROUND(CAST(bs.longTermDebt AS REAL) / CAST(bs.totalStockholdersEquity AS REAL), 2) AS debt_to_equity FROM balance_sheets_quarterly bs JOIN sp500_constituents sc ON bs.symbol = sc.symbol WHERE bs.date >= '2025-01-01' AND bs.totalStockholdersEquity > 0 AND bs.longTermDebt > 0 ORDER BY debt_to_equity DESC LIMIT 10",
                "category": "calculation",
                "complexity": "medium",
                "intent": "financial_ratio",
                "success_count": 28,
                "explanation": "Calculate financial ratios with division operations"
            },
            {
                "id": "medium_7",
                "question": "Top companies by free cash flow (recent quarters)",
                "sql": "SELECT sc.symbol, sc.name, cfs.freeCashFlow, cfs.date FROM cash_flow_statements_quarterly cfs JOIN sp500_constituents sc ON cfs.symbol = sc.symbol WHERE cfs.date >= '2025-01-01' AND cfs.freeCashFlow IS NOT NULL ORDER BY cfs.freeCashFlow DESC LIMIT 10",
                "category": "ranking",
                "complexity": "medium",
                "intent": "cash_flow_ranking",
                "success_count": 26,
                "explanation": "Rank by cash flow metrics with JOIN"
            },
            {
                "id": "annual_2",
                "question": "Top 5 companies by most recent annual net income",
                "sql": "SELECT sc.symbol, sc.name, isa.netIncome, isa.date FROM income_statements_annual isa JOIN sp500_constituents sc ON isa.symbol = sc.symbol WHERE isa.date >= '2024-01-01' AND isa.netIncome IS NOT NULL ORDER BY isa.netIncome DESC LIMIT 5",
                "category": "annual_ranking",
                "complexity": "medium",
                "intent": "annual_profitability",
                "success_count": 32,
                "explanation": "Rank companies by annual profitability"
            },
            {
                "id": "annual_3",
                "question": "Microsoft annual revenue growth over the past three years",
                "sql": "SELECT symbol, date, revenue, LAG(revenue, 1) OVER (ORDER BY date) as prev_year_revenue FROM income_statements_annual WHERE symbol = 'MSFT' AND date >= '2022-01-01' ORDER BY date DESC",
                "category": "annual_trend",
                "complexity": "medium",
                "intent": "growth_analysis",
                "success_count": 24,
                "explanation": "Analyze year-over-year growth using window functions"
            },
            {
                "id": "annual_5",
                "question": "Companies with the strongest equity on the latest annual balance sheet",
                "sql": "SELECT sc.symbol, sc.name, bsa.totalStockholdersEquity, bsa.date FROM balance_sheets_annual bsa JOIN sp500_constituents sc ON bsa.symbol = sc.symbol WHERE bsa.date >= '2024-01-01' AND bsa.totalStockholdersEquity > 0 ORDER BY bsa.totalStockholdersEquity DESC LIMIT 10",
                "category": "annual_financial_strength",
                "complexity": "medium",
                "intent": "financial_strength",
                "success_count": 29,
                "explanation": "Identify financially strong companies by equity"
            },
            {
                "id": "annual_7",
                "question": "Companies with the highest annual free cash flow",
                "sql": "SELECT sc.symbol, sc.name, cfsa.freeCashFlow, cfsa.date, sc.sector FROM cash_flow_statements_annual cfsa JOIN sp500_constituents sc ON cfsa.symbol = sc.symbol WHERE cfsa.date >= '2024-01-01' AND cfsa.freeCashFlow IS NOT NULL ORDER BY cfsa.freeCashFlow DESC LIMIT 10",
                "category": "annual_cash_ranking",
                "complexity": "medium",
                "intent": "cash_generation",
                "success_count": 31,
                "explanation": "Rank companies by cash generation ability"
            }
        ]
        
        # Complex queries with advanced SQL
        complex_queries = [
            {
                "id": "complex_1",
                "question": "Companies showing sequential revenue growth across the last three quarters",
                "sql": """WITH quarterly_revenue AS (
                    SELECT symbol, date, revenue, 
                           LAG(revenue, 1) OVER (PARTITION BY symbol ORDER BY date) AS prev_quarter_1, 
                           LAG(revenue, 2) OVER (PARTITION BY symbol ORDER BY date) AS prev_quarter_2 
                    FROM income_statements_quarterly 
                    WHERE date >= '2024-06-01'
                ) 
                SELECT qr.symbol, sc.name, qr.revenue, qr.prev_quarter_1, qr.prev_quarter_2, qr.date 
                FROM quarterly_revenue qr 
                JOIN sp500_constituents sc ON qr.symbol = sc.symbol 
                WHERE qr.revenue > qr.prev_quarter_1 
                  AND qr.prev_quarter_1 > qr.prev_quarter_2 
                  AND qr.prev_quarter_1 IS NOT NULL 
                  AND qr.prev_quarter_2 IS NOT NULL 
                ORDER BY qr.revenue DESC""",
                "category": "trend_analysis",
                "complexity": "complex",
                "intent": "growth_trend",
                "success_count": 45,
                "explanation": "Identify revenue growth trends using CTEs and window functions"
            },
            {
                "id": "complex_2",
                "question": "Companies with high revenue relative to long-term debt",
                "sql": """WITH latest_financials AS (
                    SELECT i.symbol, i.revenue, b.longTermDebt 
                    FROM income_statements_quarterly i 
                    JOIN balance_sheets_quarterly b ON i.symbol = b.symbol 
                      AND ABS(JULIANDAY(i.date) - JULIANDAY(b.date)) <= 93 
                    WHERE i.date >= '2025-01-01' AND b.date >= '2025-01-01'
                ) 
                SELECT sc.symbol, sc.name, lf.revenue, lf.longTermDebt, 
                       ROUND(CAST(lf.revenue AS REAL) / NULLIF(CAST(lf.longTermDebt AS REAL), 0), 2) AS revenue_to_debt_ratio 
                FROM latest_financials lf 
                JOIN sp500_constituents sc ON lf.symbol = sc.symbol 
                WHERE lf.longTermDebt > 0 
                ORDER BY revenue_to_debt_ratio DESC LIMIT 15""",
                "category": "financial_strength",
                "complexity": "complex",
                "intent": "quality_analysis",
                "success_count": 38,
                "explanation": "Complex financial analysis with multiple table joins and CTEs"
            },
            {
                "id": "complex_3",
                "question": "Year-over-year revenue growth for high-revenue companies",
                "sql": """WITH current_year AS (
                    SELECT symbol, revenue, date, 
                           ROW_NUMBER() OVER (PARTITION BY symbol ORDER BY date DESC) as rn 
                    FROM income_statements_quarterly 
                    WHERE date >= '2024-01-01'
                ), 
                previous_year AS (
                    SELECT symbol, revenue as prev_revenue 
                    FROM income_statements_quarterly 
                    WHERE date >= '2023-01-01' AND date < '2024-01-01'
                ) 
                SELECT sc.symbol, sc.name, cy.revenue as current_revenue, py.prev_revenue, 
                       ROUND(((cy.revenue - py.prev_revenue) * 100.0 / py.prev_revenue), 2) AS yoy_growth 
                FROM current_year cy 
                JOIN previous_year py ON cy.symbol = py.symbol 
                JOIN sp500_constituents sc ON cy.symbol = sc.symbol 
                WHERE cy.rn = 1 AND py.prev_revenue > 0 
                ORDER BY current_revenue DESC LIMIT 10""",
                "category": "growth_analysis",
                "complexity": "complex",
                "intent": "yoy_growth",
                "success_count": 42,
                "explanation": "Year-over-year growth calculation with multiple CTEs"
            },
            {
                "id": "complex_4",
                "question": "Companies where operating cash flow exceeds net income by over 20%",
                "sql": """SELECT sc.symbol, sc.name, cfs.operatingCashFlow, isq.netIncome, 
                         ROUND((cfs.operatingCashFlow - isq.netIncome) * 100.0 / ABS(isq.netIncome), 2) AS cash_premium_pct 
                  FROM cash_flow_statements_quarterly cfs 
                  JOIN income_statements_quarterly isq ON cfs.symbol = isq.symbol AND cfs.date = isq.date 
                  JOIN sp500_constituents sc ON cfs.symbol = sc.symbol 
                  WHERE cfs.date >= '2025-01-01' 
                    AND isq.netIncome != 0 
                    AND cfs.operatingCashFlow > isq.netIncome * 1.2 
                  ORDER BY cash_premium_pct DESC LIMIT 15""",
                "category": "quality_analysis",
                "complexity": "complex",
                "intent": "earnings_quality",
                "success_count": 40,
                "explanation": "Identify high-quality earnings with cash flow analysis and multiple joins"
            }
        ]
        
        # Add edge case examples
        edge_cases = [
            {
                "id": "edge_1",
                "question": "Attempt to retrieve a non-existent company's record",
                "sql": "SELECT symbol, name, sector FROM sp500_constituents WHERE symbol = 'XXXX'",
                "category": "error_handling",
                "complexity": "edge",
                "intent": "error_case",
                "success_count": 5,
                "explanation": "Handle non-existent companies gracefully"
            },
            {
                "id": "edge_2",
                "question": "Query prices for a date likely outside the available range (2000-01-01)",
                "sql": "SELECT symbol, date, close FROM historical_prices WHERE date = '2000-01-01'",
                "category": "date_handling",
                "complexity": "edge",
                "intent": "historical_data",
                "success_count": 3,
                "explanation": "Handle dates outside data range"
            },
            {
                "id": "edge_3",
                "question": "Compute square roots of revenue values",
                "sql": "SELECT symbol, revenue, SQRT(revenue) as sqrt_revenue FROM income_statements_quarterly WHERE revenue IS NOT NULL ORDER BY date DESC LIMIT 10",
                "category": "math_functions",
                "complexity": "edge",
                "intent": "mathematical",
                "success_count": 8,
                "explanation": "Handle mathematical functions correctly"
            }
        ]
        
        # Combine all examples
        all_examples = simple_queries + medium_queries + complex_queries + edge_cases
        
        # Add metadata
        for example in all_examples:
            example.update({
                "last_used": datetime.now().isoformat(),
                "created_date": datetime.now().isoformat(),
                "version": "1.0"
            })
        
        # Save to file
        self.save_training_data(all_examples)
        return all_examples

    def save_training_data(self, data: List[Dict]):
        """Save training data to file"""
        try:
            with open(self.training_data_path, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving training data: {e}")

    def load_query_stats(self) -> Dict:
        """Load query execution statistics"""
        try:
            if os.path.exists(self.stats_path):
                with open(self.stats_path, 'r') as f:
                    return json.load(f)
        except Exception as e:
            logger.error(f"Error loading query stats: {e}")
        return {"total_queries": 0, "successful_queries": 0, "category_stats": {}}

    def save_query_stats(self):
        """Save query execution statistics"""
        try:
            with open(self.stats_path, 'w') as f:
                json.dump(self.query_stats, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving query stats: {e}")

    def initialize_comprehensive_rag(self):
        """Initialize comprehensive RAG system"""
        try:
            if (os.path.exists(self.vectorizer_path) and 
                os.path.exists(self.embeddings_path) and 
                len(self.training_examples) > 0):
                
                with open(self.vectorizer_path, 'rb') as f:
                    self.vectorizer = pickle.load(f)
                with open(self.embeddings_path, 'rb') as f:
                    self.example_embeddings = pickle.load(f)
                    
                # Check if we need to rebuild (new examples added)
                if self.example_embeddings.shape[0] != len(self.training_examples):
                    self.build_comprehensive_index()
                    
                logger.info("Loaded existing comprehensive RAG components")
            else:
                self.build_comprehensive_index()
        except Exception as e:
            logger.error(f"Error initializing comprehensive RAG: {e}")
            self.build_comprehensive_index()

    def build_comprehensive_index(self):
        """Build comprehensive vector index"""
        try:
            if not self.training_examples:
                logger.warning("No training examples for RAG index")
                return

            # Enhanced feature extraction for financial queries
            questions = [ex['question'] for ex in self.training_examples]
            
            # Create comprehensive TF-IDF vectorizer
            self.vectorizer = TfidfVectorizer(
                max_features=10000,  # Increased for better coverage
                stop_words='english',
                ngram_range=(1, 4),  # Include 4-grams for better phrase matching
                lowercase=True,
                min_df=1,  # Include rare terms
                max_df=0.95,  # Remove very common terms
                analyzer='word',
                token_pattern=r'\b[A-Za-z][A-Za-z0-9]*\b'  # Include financial symbols
            )
            
            self.example_embeddings = self.vectorizer.fit_transform(questions)
            
            # Save components
            with open(self.vectorizer_path, 'wb') as f:
                pickle.dump(self.vectorizer, f)
            with open(self.embeddings_path, 'wb') as f:
                pickle.dump(self.example_embeddings, f)
                
            logger.info(f"Built comprehensive RAG index with {len(questions)} examples")
            
        except Exception as e:
            logger.error(f"Error building comprehensive index: {e}")

    def find_relevant_examples(self, query: str, top_k: int = 8) -> List[Dict]:
        """Find most relevant examples using advanced semantic matching"""
        try:
            if not self.vectorizer or self.example_embeddings is None:
                logger.warning("RAG system not initialized")
                return self.training_examples[:top_k]
            
            # Transform query
            query_embedding = self.vectorizer.transform([query])
            
            # Calculate similarities
            similarities = cosine_similarity(query_embedding, self.example_embeddings)[0]
            
            # Get top-k indices
            top_indices = np.argsort(similarities)[::-1][:top_k * 2]  # Get more candidates
            
            # Apply intelligent filtering
            candidates = []
            for idx in top_indices:
                if similarities[idx] > 0.01:  # Minimum similarity threshold
                    example = self.training_examples[idx].copy()
                    example['similarity_score'] = float(similarities[idx])
                    candidates.append(example)
            
            # Enhanced ranking considering multiple factors
            ranked_examples = self.rank_examples_intelligently(query, candidates[:top_k])
            
            logger.info(f"Found {len(ranked_examples)} relevant examples")
            return ranked_examples
            
        except Exception as e:
            logger.error(f"Error finding relevant examples: {e}")
            return self.training_examples[:top_k]

    def rank_examples_intelligently(self, query: str, candidates: List[Dict]) -> List[Dict]:
        """Intelligent ranking considering multiple factors"""
        query_lower = query.lower()
        
        # Boost factors
        for candidate in candidates:
            base_score = candidate['similarity_score']
            boost = 1.0
            
            # Success rate boost
            success_count = candidate.get('success_count', 1)
            boost *= (1.0 + min(success_count / 50.0, 0.5))  # Max 50% boost for successful queries
            
            # Category matching boost
            if self.detect_query_category(query) == candidate.get('category', ''):
                boost *= 1.3
            
            # Intent matching boost  
            if self.detect_query_intent(query) == candidate.get('intent', ''):
                boost *= 1.2
            
            # Complexity preference (prefer similar complexity)
            query_complexity = self.estimate_query_complexity(query)
            if query_complexity == candidate.get('complexity', 'simple'):
                boost *= 1.15
            
            # Recency boost (slightly prefer recently successful examples)
            try:
                last_used = datetime.fromisoformat(candidate.get('last_used', '2024-01-01T00:00:00'))
                days_since_used = (datetime.now() - last_used).days
                if days_since_used < 30:
                    boost *= 1.1
            except:
                pass
            
            candidate['final_score'] = base_score * boost
        
        # Sort by final score
        candidates.sort(key=lambda x: x['final_score'], reverse=True)
        return candidates

    def detect_query_category(self, query: str) -> str:
        """Detect query category from text"""
        query_lower = query.lower()
        
        if any(word in query_lower for word in ['price', 'stock', 'close', 'open', 'trading']):
            return 'stock_price'
        elif any(word in query_lower for word in ['revenue', 'income', 'profit', 'earnings']):
            return 'financial_metrics'
        elif any(word in query_lower for word in ['assets', 'liabilities', 'equity', 'balance']):
            return 'balance_sheet'
        elif any(word in query_lower for word in ['cash flow', 'operating cash', 'free cash']):
            return 'cash_flow'
        elif any(word in query_lower for word in ['compare', 'vs', 'versus']):
            return 'comparison'
        elif any(word in query_lower for word in ['top', 'highest', 'best', 'rank']):
            return 'ranking'
        elif any(word in query_lower for word in ['sector', 'industry', 'companies']):
            return 'company_info'
        elif any(word in query_lower for word in ['growth', 'trend', 'change']):
            return 'trend_analysis'
        else:
            return 'general'

    def detect_query_intent(self, query: str) -> str:
        """Detect specific intent from query"""
        query_lower = query.lower()
        
        if 'latest' in query_lower or 'recent' in query_lower:
            return 'latest_data'
        elif 'highest' in query_lower or 'top' in query_lower:
            return 'ranking'
        elif 'growth' in query_lower or 'increase' in query_lower:
            return 'growth_analysis'
        elif 'compare' in query_lower or 'vs' in query_lower:
            return 'comparison'
        elif 'margin' in query_lower or 'ratio' in query_lower:
            return 'financial_ratio'
        elif 'annual' in query_lower or 'yearly' in query_lower:
            return 'annual_analysis'
        elif 'quarterly' in query_lower or 'q1' in query_lower or 'q2' in query_lower:
            return 'quarterly_analysis'
        else:
            return 'general'

    def estimate_query_complexity(self, query: str) -> str:
        """Estimate query complexity based on keywords"""
        query_lower = query.lower()
        
        complex_indicators = [
            'growth over', 'year over year', 'trend', 'compare.*over time',
            'increasing', 'decreasing', 'correlation', 'multiple quarters',
            'best combination', 'where.*exceeds', 'ratio.*and', 'both.*and'
        ]
        
        medium_indicators = [
            'top', 'average', 'highest', 'lowest', 'compare', 'vs',
            'margin', 'ratio', 'sector', 'companies with'
        ]
        
        if any(re.search(indicator, query_lower) for indicator in complex_indicators):
            return 'complex'
        elif any(indicator in query_lower for indicator in medium_indicators):
            return 'medium'
        else:
            return 'simple'

    def generate_sql_with_comprehensive_rag(self, user_question: str) -> str:
        """Generate SQL using comprehensive RAG system"""
        try:
            # Find most relevant examples
            relevant_examples = self.find_relevant_examples(user_question, top_k=6)
            
            # Create enhanced context-aware prompt
            schema_text = self.get_enhanced_schema_text()
            examples_text = self.format_examples_for_prompt(relevant_examples)
            context_info = self.get_query_context(user_question)
            
            prompt = f"""You are an expert Financial SQL analyst with access to comprehensive S&P 500 financial database.

{schema_text}

CONTEXT ANALYSIS:
- Query Category: {context_info['category']}
- Intent: {context_info['intent']}  
- Estimated Complexity: {context_info['complexity']}
- Date References: {context_info.get('dates', 'None detected')}

RELEVANT EXAMPLES (ranked by similarity and success rate):
{examples_text}

USER QUESTION: "{user_question}"

CRITICAL RULES:
1. Use ONLY exact table/column names from schema above
2. For financial ratios, always use NULLIF to prevent division by zero
3. Use CAST(column AS REAL) for mathematical operations to ensure proper division
4. For "latest" or "recent", use ORDER BY date DESC LIMIT
5. For quarters: Q1=Jan-Mar, Q2=Apr-Jun, Q3=Jul-Sep, Q4=Oct-Dec
6. Join tables on 'symbol' column when combining data
7. Handle NULL values appropriately in all calculations
8. Use proper date formats: 'YYYY-MM-DD'
9. For revenue/income questions, avoid unnecessary UNION statements
10. Focus on the specific question asked - don't over-complicate

Generate ONLY the SQL query, no explanations:"""

            response = openai.ChatCompletion.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a SQL expert specializing in financial data analysis. Return only valid SQLite queries using the exact schema provided. Focus on accuracy and efficiency."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=1000
            )

            sql_query = response["choices"][0]["message"]["content"].strip()
            return self.clean_and_validate_sql(sql_query)
            
        except Exception as e:
            logger.error(f"Error generating SQL with comprehensive RAG: {e}")
            return None

    def get_query_context(self, query: str) -> Dict[str, Any]:
        """Extract comprehensive context from query"""
        return {
            'category': self.detect_query_category(query),
            'intent': self.detect_query_intent(query),
            'complexity': self.estimate_query_complexity(query),
            'dates': self.extract_dates_from_query(query),
            'symbols': self.extract_symbols_from_query(query),
            'financial_terms': self.extract_financial_terms(query)
        }

    def extract_dates_from_query(self, query: str) -> List[str]:
        """Extract date references from query"""
        date_patterns = [
            r'\d{4}-\d{2}-\d{2}',  # YYYY-MM-DD
            r'\d{4}',              # Year
            r'Q[1-4]\s+\d{4}',     # Q1 2025
            r'[Qq]uarter\s+[1-4]',  # quarter 1
            r'last\s+\d+\s+quarters?',
            r'last\s+\d+\s+years?'
        ]
        
        dates = []
        for pattern in date_patterns:
            matches = re.findall(pattern, query, re.IGNORECASE)
            dates.extend(matches)
        
        return dates

    def extract_symbols_from_query(self, query: str) -> List[str]:
        """Extract stock symbols from query"""
        # Common company names to symbols mapping
        company_symbols = {
            'apple': 'AAPL', 'microsoft': 'MSFT', 'amazon': 'AMZN',
            'google': 'GOOGL', 'tesla': 'TSLA', 'meta': 'META',
            'nvidia': 'NVDA', 'walmart': 'WMT', 'disney': 'DIS'
        }
        
        symbols = []
        query_lower = query.lower()
        
        # Direct symbol matches (2-5 uppercase letters)
        symbol_matches = re.findall(r'\b[A-Z]{2,5}\b', query)
        symbols.extend(symbol_matches)
        
        # Company name matches
        for company, symbol in company_symbols.items():
            if company in query_lower:
                symbols.append(symbol)
        
        return list(set(symbols))

    def extract_financial_terms(self, query: str) -> List[str]:
        """Extract financial terms from query"""
        financial_terms = [
            'revenue', 'profit', 'margin', 'ratio', 'cash flow', 'assets',
            'equity', 'debt', 'income', 'earnings', 'growth', 'volatility'
        ]
        
        found_terms = []
        query_lower = query.lower()
        
        for term in financial_terms:
            if term in query_lower:
                found_terms.append(term)
        
        return found_terms

    def format_examples_for_prompt(self, examples: List[Dict]) -> str:
        """Format examples for inclusion in prompt"""
        formatted = ""
        
        for i, ex in enumerate(examples, 1):
            similarity = ex.get('similarity_score', 0)
            success_count = ex.get('success_count', 0)
            
            formatted += f"""
Example {i} (Similarity: {similarity:.3f}, Success: {success_count}):
Q: {ex['question']}
SQL: {ex['sql']}
Category: {ex.get('category', 'unknown')} | Complexity: {ex.get('complexity', 'simple')}
"""
        
        return formatted

    def get_enhanced_schema_text(self) -> str:
        """Get enhanced schema description"""
        return """
DATABASE SCHEMA (SQLite):

Table: historical_prices
- symbol TEXT, date DATE, open REAL, high REAL, low REAL, close REAL, volume INTEGER, change REAL, changePercent REAL

Table: sp500_constituents  
- symbol TEXT PRIMARY KEY, name TEXT, sector TEXT, subSector TEXT, headQuarter TEXT, dateFirstAdded DATE

Table: income_statements_quarterly
- symbol TEXT, date DATE, revenue BIGINT, costOfRevenue BIGINT, grossProfit BIGINT, operatingExpenses BIGINT, operatingIncome BIGINT, netIncome BIGINT, eps REAL

Table: income_statements_annual
- symbol TEXT, date DATE, revenue BIGINT, costOfRevenue BIGINT, grossProfit BIGINT, operatingExpenses BIGINT, operatingIncome BIGINT, netIncome BIGINT, eps REAL

Table: balance_sheets_quarterly
- symbol TEXT, date DATE, totalAssets BIGINT, currentAssets BIGINT, totalLiabilities BIGINT, currentLiabilities BIGINT, totalStockholdersEquity BIGINT, longTermDebt BIGINT

Table: balance_sheets_annual  
- symbol TEXT, date DATE, totalAssets BIGINT, currentAssets BIGINT, totalLiabilities BIGINT, currentLiabilities BIGINT, totalStockholdersEquity BIGINT, longTermDebt BIGINT

Table: cash_flow_statements_quarterly
- symbol TEXT, date DATE, operatingCashFlow BIGINT, investingCashFlow BIGINT, financingCashFlow BIGINT, freeCashFlow BIGINT

Table: cash_flow_statements_annual
- symbol TEXT, date DATE, operatingCashFlow BIGINT, investingCashFlow BIGINT, financingCashFlow BIGINT, freeCashFlow BIGINT

IMPORTANT NOTES:
- Always JOIN with sp500_constituents using symbol to get company names
- Use CAST(column AS REAL) for division operations
- Recent data available through 2025-06-30
- Financial values are in actual dollars (not millions/billions)
"""

    def clean_and_validate_sql(self, sql_query: str) -> str:
        """Clean and validate SQL query"""
        # Remove code block markers
        sql_query = re.sub(r'```sql\s*', '', sql_query, flags=re.IGNORECASE)
        sql_query = re.sub(r'```\s*', '', sql_query)
        
        # Clean whitespace
        sql_query = sql_query.strip()
        
        # Ensure semicolon
        if not sql_query.endswith(';'):
            sql_query += ';'
        
        # Basic validation
        if not sql_query.lower().startswith('select'):
            logger.warning("Generated query doesn't start with SELECT")
            return None
            
        return sql_query

    def learn_from_execution(self, question: str, sql: str, success: bool, execution_result: Dict = None):
        """Learn from query execution results with comprehensive feedback"""
        try:
            self.query_stats['total_queries'] += 1
            
            if success:
                self.query_stats['successful_queries'] += 1
                
                # Update category statistics
                category = self.detect_query_category(question)
                if category not in self.query_stats['category_stats']:
                    self.query_stats['category_stats'][category] = {'successful': 0, 'total': 0}
                
                self.query_stats['category_stats'][category]['successful'] += 1
                self.query_stats['category_stats'][category]['total'] += 1
                
                # Check if similar example exists
                existing_idx = self.find_similar_example(question, sql)
                
                if existing_idx >= 0:
                    # Update existing example
                    self.training_examples[existing_idx]['success_count'] += 1
                    self.training_examples[existing_idx]['last_used'] = datetime.now().isoformat()
                else:
                    # Add new successful example
                    new_example = {
                        "id": f"learned_{len(self.training_examples)}",
                        "question": question,
                        "sql": sql,
                        "category": self.detect_query_category(question),
                        "complexity": self.estimate_query_complexity(question),
                        "intent": self.detect_query_intent(question),
                        "success_count": 1,
                        "last_used": datetime.now().isoformat(),
                        "created_date": datetime.now().isoformat(),
                        "explanation": "Learned from successful execution",
                        "version": "learned"
                    }
                    
                    self.training_examples.append(new_example)
                    
                    # Rebuild index periodically
                    if len(self.training_examples) % 20 == 0:
                        self.build_comprehensive_index()
                
                # Save updated data
                self.save_training_data(self.training_examples)
                self.save_query_stats()
                
                logger.info(f"Learned from successful query: {question[:50]}...")
            else:
                # Track failed queries for analysis
                category = self.detect_query_category(question)
                if category not in self.query_stats['category_stats']:
                    self.query_stats['category_stats'][category] = {'successful': 0, 'total': 0}
                
                self.query_stats['category_stats'][category]['total'] += 1
                
        except Exception as e:
            logger.error(f"Error learning from execution: {e}")

    def find_similar_example(self, question: str, sql: str) -> int:
        """Find if similar example exists using semantic similarity"""
        try:
            if not self.vectorizer:
                return -1
            
            question_embedding = self.vectorizer.transform([question])
            similarities = cosine_similarity(question_embedding, self.example_embeddings)[0]
            
            # Find most similar example
            max_sim_idx = np.argmax(similarities)
            max_similarity = similarities[max_sim_idx]
            
            # Check if similarity is high enough to consider it the same query
            if max_similarity > 0.85:
                return max_sim_idx
            
            return -1
            
        except Exception as e:
            logger.error(f"Error finding similar example: {e}")
            return -1

    def get_comprehensive_statistics(self) -> Dict[str, Any]:
        """Get comprehensive system statistics"""
        try:
            stats = {
                "training_data": {
                    "total_examples": len(self.training_examples),
                    "by_complexity": self.get_complexity_distribution(),
                    "by_category": self.get_category_distribution(),
                    "by_intent": self.get_intent_distribution(),
                    "success_rate_distribution": self.get_success_rate_distribution()
                },
                "query_execution": self.query_stats,
                "rag_system": {
                    "vectorizer_active": self.vectorizer is not None,
                    "embeddings_shape": self.example_embeddings.shape if self.example_embeddings is not None else None,
                    "vocabulary_size": len(self.vectorizer.vocabulary_) if self.vectorizer else 0
                },
                "top_performing_categories": self.get_top_performing_categories(),
                "recent_learning": self.get_recent_learning_stats()
            }
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting statistics: {e}")
            return {"error": str(e)}

    def get_complexity_distribution(self) -> Dict[str, int]:
        """Get distribution by complexity"""
        dist = defaultdict(int)
        for ex in self.training_examples:
            complexity = ex.get('complexity', 'simple')
            dist[complexity] += 1
        return dict(dist)

    def get_category_distribution(self) -> Dict[str, int]:
        """Get distribution by category"""
        dist = defaultdict(int)
        for ex in self.training_examples:
            category = ex.get('category', 'general')
            dist[category] += 1
        return dict(dist)

    def get_intent_distribution(self) -> Dict[str, int]:
        """Get distribution by intent"""
        dist = defaultdict(int)
        for ex in self.training_examples:
            intent = ex.get('intent', 'general')
            dist[intent] += 1
        return dict(dist)

    def get_success_rate_distribution(self) -> Dict[str, int]:
        """Get distribution by success rate ranges"""
        ranges = {"0": 0, "1-5": 0, "6-15": 0, "16-30": 0, "30+": 0}
        
        for ex in self.training_examples:
            success_count = ex.get('success_count', 0)
            if success_count == 0:
                ranges["0"] += 1
            elif success_count <= 5:
                ranges["1-5"] += 1
            elif success_count <= 15:
                ranges["6-15"] += 1
            elif success_count <= 30:
                ranges["16-30"] += 1
            else:
                ranges["30+"] += 1
                
        return ranges

    def get_top_performing_categories(self) -> List[Dict]:
        """Get top performing categories by success rate"""
        category_performance = {}
        
        for category, stats in self.query_stats.get('category_stats', {}).items():
            if stats['total'] > 0:
                success_rate = stats['successful'] / stats['total']
                category_performance[category] = {
                    'success_rate': success_rate,
                    'total_queries': stats['total'],
                    'successful_queries': stats['successful']
                }
        
        # Sort by success rate
        sorted_categories = sorted(category_performance.items(), 
                                 key=lambda x: x[1]['success_rate'], 
                                 reverse=True)
        
        return [{"category": cat, **stats} for cat, stats in sorted_categories[:10]]

    def get_recent_learning_stats(self) -> Dict[str, Any]:
        """Get statistics about recent learning"""
        try:
            recent_examples = [ex for ex in self.training_examples 
                             if ex.get('version') == 'learned']
            
            last_30_days = datetime.now() - timedelta(days=30)
            recent_activity = [ex for ex in self.training_examples
                             if datetime.fromisoformat(ex.get('last_used', '2024-01-01T00:00:00')) > last_30_days]
            
            return {
                "learned_examples": len(recent_examples),
                "active_examples_last_30_days": len(recent_activity),
                "average_success_count": sum(ex.get('success_count', 0) for ex in self.training_examples) / len(self.training_examples) if self.training_examples else 0
            }
            
        except Exception as e:
            logger.error(f"Error getting recent learning stats: {e}")
            return {"error": str(e)}

    def get_schema(self) -> Dict[str, Any]:
        """Get database schema information"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            schema = {}

            # Get all tables
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            tables = [row[0] for row in cursor.fetchall()]

            for table in tables:
                cursor.execute(f"PRAGMA table_info({table});")
                columns_info = cursor.fetchall()
                
                cursor.execute(f"SELECT COUNT(*) FROM {table};")
                row_count = cursor.fetchone()[0]
                
                schema[table] = {
                    'columns': [
                        {
                            'name': col[1],
                            'type': col[2],
                            'nullable': not col[3],
                            'primary_key': bool(col[5])
                        } for col in columns_info
                    ],
                    'row_count': row_count
                }

            conn.close()
            return schema
        except Exception as e:
            logger.error(f"Error loading schema: {e}")
            return {}

# Flask Application Integration
app = Flask(__name__)
CORS(app)

# Global variable to hold our AI instance
enhanced_ai = None

def initialize_ai():
    """Initialize the AI system"""
    global enhanced_ai
    try:
        # Ensure OpenAI is configured before initializing components that use it
        configure_openai()
        enhanced_ai = ComprehensiveFinancialRAG()
        logger.info("AI system initialized successfully")
        return True
    except Exception as e:
        logger.error(f"Failed to initialize AI system: {e}")
        return False

def execute_sql_query(query, original_question):
    """Execute SQL query with error handling"""
    try:
        conn = sqlite3.connect("financials.db")
        cursor = conn.cursor()
        
        # Execute the query
        cursor.execute(query)
        results = cursor.fetchall()
        
        # Get column names
        column_names = [description[0] for description in cursor.description]
        
        # Convert to list of dictionaries
        formatted_results = []
        for row in results:
            row_dict = {}
            for i, value in enumerate(row):
                row_dict[column_names[i]] = value
            formatted_results.append(row_dict)
        
        conn.close()
        
        return {
            "success": True,
            "data": formatted_results,
            "column_names": column_names,
            "row_count": len(formatted_results),
            "message": f"Query executed successfully. Found {len(formatted_results)} results."
        }
        
    except sqlite3.Error as e:
        return {
            "success": False,
            "error": f"Database error: {str(e)}",
            "data": [],
            "column_names": [],
            "row_count": 0
        }
    except Exception as e:
        return {
            "success": False,
            "error": f"Execution error: {str(e)}",
            "data": [],
            "column_names": [],
            "row_count": 0
        }

@app.route('/')
def index():
    """Serve the main page"""
    return render_template('index.html')

@app.route('/api/query', methods=['POST'])
def process_query():
    """Process natural language query and return SQL + results"""
    try:
        data = request.get_json()
        if not data or 'question' not in data:
            return jsonify({"error": "No question provided"}), 400
        
        question = data['question'].strip()
        logger.info(f"Processing query: {question}")
        
        if not enhanced_ai:
            return jsonify({"error": "AI system not initialized"}), 500
        
        # Generate SQL using your existing method
        sql_query = enhanced_ai.generate_sql_with_comprehensive_rag(question)
        
        if not sql_query:
            return jsonify({"error": "Failed to generate SQL query"}), 400
        
        # Execute the query
        execution_result = execute_sql_query(sql_query, question)
        
        # Learn from the execution
        enhanced_ai.learn_from_execution(
            question=question,
            sql=sql_query,
            success=execution_result['success'],
            execution_result=execution_result
        )
        
        # Get additional context
        context = enhanced_ai.get_query_context(question)
        
        # Build frontend-friendly results shape (columns + array rows)
        frontend_results = None
        try:
            if execution_result.get('success'):
                columns = execution_result.get('column_names', [])
                # execution_result['data'] is a list of dicts keyed by column names
                rows = []
                for row_dict in execution_result.get('data', []):
                    rows.append([row_dict.get(col) for col in columns])
                frontend_results = {
                    "success": True,
                    "row_count": execution_result.get('row_count', len(rows)),
                    "columns": columns,
                    "data": rows
                }
            else:
                frontend_results = {
                    "success": False,
                    "error": execution_result.get('error', 'Unknown error'),
                    "row_count": 0,
                    "columns": [],
                    "data": []
                }
        except Exception as _:
            # Fallback minimal shape
            frontend_results = {"success": False, "error": "Result formatting error"}
        
        response = {
            "question": question,
            "sql_query": sql_query,
            "execution_result": execution_result,
            "results": frontend_results,
            "context": context,
            "timestamp": datetime.now().isoformat()
        }
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Error processing query: {e}")
        logger.error(traceback.format_exc())
        return jsonify({
            "error": str(e),
            "sql_query": "",
            "execution_result": {
                "success": False,
                "error": str(e),
                "data": [],
                "column_names": [],
                "row_count": 0
            }
        }), 500

@app.route('/api/enhanced-query', methods=['POST'])
def process_enhanced_query():
    """Enhanced query processing endpoint"""
    try:
        data = request.get_json()
        if not data or 'question' not in data:
            return jsonify({"error": "No question provided"}), 400
        
        question = data['question'].strip()
        logger.info(f"Processing enhanced query: {question}")
        
        if not enhanced_ai:
            return jsonify({"error": "AI system not initialized"}), 500
        
        # Generate SQL using comprehensive RAG
        sql_query = enhanced_ai.generate_sql_with_comprehensive_rag(question)
        
        if not sql_query:
            return jsonify({"error": "Failed to generate SQL query"}), 400
        
        # Execute query
        execution_result = execute_sql_query(sql_query, question)
        
        # Learn from results
        enhanced_ai.learn_from_execution(
            question=question,
            sql=sql_query,
            success=execution_result.get('success', False),
            execution_result=execution_result
        )
        
        # Get query context for response
        context = enhanced_ai.get_query_context(question)
        
        # Build frontend-friendly results shape
        frontend_results = None
        try:
            if execution_result.get('success'):
                columns = execution_result.get('column_names', [])
                rows = []
                for row_dict in execution_result.get('data', []):
                    rows.append([row_dict.get(col) for col in columns])
                frontend_results = {
                    "success": True,
                    "row_count": execution_result.get('row_count', len(rows)),
                    "columns": columns,
                    "data": rows
                }
            else:
                frontend_results = {
                    "success": False,
                    "error": execution_result.get('error', 'Unknown error'),
                    "row_count": 0,
                    "columns": [],
                    "data": []
                }
        except Exception as _:
            frontend_results = {"success": False, "error": "Result formatting error"}
        
        return jsonify({
            "question": question,
            "sql_query": sql_query,
            "results": frontend_results,
            "context": context,
            "system_stats": enhanced_ai.get_comprehensive_statistics(),
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error in enhanced query processing: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/training-stats', methods=['GET'])
def get_training_statistics():
    """Get comprehensive training statistics"""
    try:
        if not enhanced_ai:
            return jsonify({"error": "AI system not initialized"}), 500
        
        stats = enhanced_ai.get_comprehensive_statistics()
        return jsonify(stats)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/add-training-example', methods=['POST'])
def add_training_example():
    """Manually add training example"""
    try:
        if not enhanced_ai:
            return jsonify({"error": "AI system not initialized"}), 500
            
        data = request.get_json()
        if not data or 'question' not in data or 'sql' not in data:
            return jsonify({"error": "Missing question or sql in request"}), 400
            
        enhanced_ai.learn_from_execution(
            question=data['question'],
            sql=data['sql'], 
            success=True
        )
        return jsonify({"message": "Training example added successfully"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/schema', methods=['GET'])
def get_schema():
    """Get database schema information"""
    try:
        if not enhanced_ai:
            return jsonify({"error": "AI system not initialized"}), 500
            
        schema = enhanced_ai.get_schema()
        return jsonify(schema)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "ai_initialized": enhanced_ai is not None,
        "timestamp": datetime.now().isoformat()
    })

@app.errorhandler(404)
def not_found_error(error):
    return jsonify({"error": "Endpoint not found"}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({"error": "Internal server error"}), 500

if __name__ == '__main__':
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Initialize the AI system
    print("Initializing AI system...")
    configured = configure_openai()
    if not configured:
        print("WARNING: OpenAI API key not set. Set OPENAI_API_KEY env var or edit OPENAI_API_KEY in app.py.")
    if initialize_ai():
        print("AI system initialized successfully!")
        print("Starting Flask server...")
        print("Access your application at: http://localhost:5000")
        
        # Run the Flask app
        app.run(
            host='0.0.0.0',
            port=5000,
            debug=True,
            threaded=True
        )
    else:
        print("Failed to initialize AI system. Please check your database and dependencies.")