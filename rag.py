import sys, os
import numpy as np
import os
import uuid
import tempfile
from typing import List, Dict, Any, Optional
from pathlib import Path
import re
import textwrap
import json
import shutil
import numpy as np
from PIL import Image
import pytesseract
from anthropic import Anthropic

# Set tesseract path
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Initialize flags
PDF2IMAGE_AVAILABLE = False
PYMUPDF_AVAILABLE = False
CHROMADB_AVAILABLE = False
SENTENCE_TRANSFORMERS_AVAILABLE = False

# Import required packages
try:
    import pdf2image
    PDF2IMAGE_AVAILABLE = True
except ImportError:
    pass

try:
    import fitz
    PYMUPDF_AVAILABLE = True
except ImportError:
    pass

try:
    import chromadb
    from chromadb.utils import embedding_functions
    CHROMADB_AVAILABLE = True
except ImportError:
    pass

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    pass

# Create a simple in-memory vector store as fallback
class SimpleVectorStore:
    """Simple in-memory vector store for fallback"""

    def __init__(self):
        self.documents = []
        self.metadatas = []
        self.ids = []

    def add(self, documents, metadatas, ids):
        self.documents.extend(documents)
        self.metadatas.extend(metadatas)
        self.ids.extend(ids)

    def query(self, query_texts, n_results=5):
        # Simple keyword matching (fallback if embeddings fail)
        results = {"ids": [[]], "documents": [[]], "metadatas": [[]], "distances": [[]]}

        if not self.documents:
            return results

        # Very basic search - just look for term frequency
        query = query_texts[0].lower()
        scores = []

        for doc in self.documents:
            # Count occurrences of query terms in document
            score = 0
            for term in query.split():
                if len(term) > 3:  # Only consider meaningful terms
                    score += doc.lower().count(term)
            scores.append(score)

        # Get indices of top n_results
        if max(scores) > 0:
            top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:n_results]

            results["ids"] = [[self.ids[i] for i in top_indices]]
            results["documents"] = [[self.documents[i] for i in top_indices]]
            results["metadatas"] = [[self.metadatas[i] for i in top_indices]]
            results["distances"] = [[1.0 - scores[i]/max(1, max(scores)) for i in top_indices]]

        return results

class PDFScreenshotProcessor:

    def __init__(self, dpi: int = 300):
        self.dpi = dpi

    def convert_pdf_to_images(self, pdf_path: str) -> List[Image.Image]:
        print(f"Converting PDF to images: {pdf_path}")

        # Try pdf2image first
        if PDF2IMAGE_AVAILABLE:
            try:
                # Convert PDF to images
                images = pdf2image.convert_from_path(
                    pdf_path,
                    dpi=self.dpi
                )
                print(f"Converted {len(images)} pages using pdf2image")
                return images
            except Exception as e:
                print(f"Error using pdf2image: {e}")
                print("Falling back to PyMuPDF...")

        # Fall back to PyMuPDF if pdf2image fails or isn't available
        if PYMUPDF_AVAILABLE:
            try:
                images = []
                pdf_document = fitz.open(pdf_path)
                for page_number in range(len(pdf_document)):
                    page = pdf_document.load_page(page_number)
                    # Render page to an image with higher resolution for better OCR
                    pix = page.get_pixmap(matrix=fitz.Matrix(self.dpi/72, self.dpi/72))
                    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                    images.append(img)
                print(f"Converted {len(images)} pages using PyMuPDF")
                return images
            except Exception as e:
                print(f"Error using PyMuPDF: {e}")

        raise ValueError("Could not convert PDF to images. Please check if the PDF is valid and not corrupted.")

    def extract_text_from_image(self, image: Image.Image) -> str:
        text = pytesseract.image_to_string(image)
        return text

    def process_pdf(self, pdf_path: str) -> List[Dict[str, Any]]:
        # Convert PDF to images
        images = self.convert_pdf_to_images(pdf_path)

        # Extract text from each image
        pages = []
        for i, image in enumerate(images):
            page_num = i + 1
            text = self.extract_text_from_image(image)

            # Save results
            pages.append({
                "page_id": f"page_{uuid.uuid4()}",
                "pdf_name": os.path.basename(pdf_path),
                "page_num": page_num,
                "text": text,
                "image": image
            })

            print(f"Processed page {page_num}/{len(images)}")

        return pages
    def extract_tables_from_pages(self, pages: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        """Extract tables from all pages"""
        table_extractor = TableExtractor()
        tables_by_page = {}
        
        for page in pages:
            page_tables = table_extractor.extract_tables_from_image(page['image'])
            if page_tables:
                # Add page information to each table
                for table in page_tables:
                    table['page_num'] = page['page_num']
                    table['pdf_name'] = page['pdf_name']
                
                tables_by_page[str(page['page_num'])] = page_tables
                print(f"Extracted {len(page_tables)} tables from page {page['page_num']}")
            
        return tables_by_page


class ChunkStrategy:
    """Strategy for chunking text from PDF pages"""

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk_text(self, text: str, metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        cleaned_text = self._clean_text(text)

        # Split into paragraphs
        paragraphs = cleaned_text.split('\n\n')
        paragraphs = [p for p in paragraphs if p.strip()]

        # Create chunks
        chunks = []
        current_chunk = ""

        for para in paragraphs:
            # If adding this paragraph would exceed chunk size, save current chunk
            if len(current_chunk) + len(para) > self.chunk_size and current_chunk:
                # Save current chunk
                chunk_id = f"chunk_{uuid.uuid4()}"
                chunks.append({
                    "chunk_id": chunk_id,
                    "text": current_chunk.strip(),
                    "metadata": {**metadata, "chunk_id": chunk_id}
                })

                # Start new chunk with overlap
                words = current_chunk.split()
                overlap_text = " ".join(words[-self.chunk_overlap:]) if len(words) > self.chunk_overlap else current_chunk
                current_chunk = overlap_text + "\n\n" + para
            else:
                # Add paragraph to current chunk
                if current_chunk:
                    current_chunk += "\n\n" + para
                else:
                    current_chunk = para

        # Add the final chunk if it's not empty
        if current_chunk:
            chunk_id = f"chunk_{uuid.uuid4()}"
            chunks.append({
                "chunk_id": chunk_id,
                "text": current_chunk.strip(),
                "metadata": {**metadata, "chunk_id": chunk_id}
            })

        return chunks

    def _clean_text(self, text: str) -> str:
        cleaned = re.sub(r'\s+', ' ', text)

        # Remove common OCR artifacts
        cleaned = cleaned.replace('|', 'I')  # Common OCR mistake

        # Fix broken paragraphs (lines ending with hyphen)
        cleaned = re.sub(r'(\w+)-\s*\n\s*(\w+)', r'\1\2', cleaned)

        # Remove nonprintable characters
        cleaned = ''.join(c for c in cleaned if c.isprintable() or c in '\n\t')

        # Wrap long lines
        wrapped = []
        for line in cleaned.split('\n'):
            if len(line) > 100:  # Arbitrary threshold for long lines
                wrapped.extend(textwrap.wrap(line, width=100))
            else:
                wrapped.append(line)

        return '\n'.join(wrapped)

class VectorDatabaseManager:
    """Manager for the vector database"""

    def __init__(self, collection_name: str = "pdf_screenshots"):
        if not CHROMADB_AVAILABLE:
            print("Using simple in-memory vector store")
            self.collection = SimpleVectorStore()
            return

        # Create persistence directory with unique name
        self.persist_directory = f"chroma_db_{collection_name}"

        # Clean up any existing database files
        if os.path.exists(self.persist_directory):
            try:
                shutil.rmtree(self.persist_directory)
                print(f"Removed existing database directory: {self.persist_directory}")
            except Exception as e:
                print(f"Warning: Could not remove existing directory: {e}")

        # Initialize ChromaDB client with persistence
        try:
            self.client = chromadb.PersistentClient(path=self.persist_directory)
            print(f"Initialized ChromaDB with persistence at: {self.persist_directory}")
        except Exception as e:
            print(f"Error creating persistent client: {e}")
            print("Falling back to in-memory client")
            try:
                self.client = chromadb.Client()
                print("Created in-memory ChromaDB client")
            except Exception as e2:
                print(f"Error creating in-memory client: {e2}")
                print("Falling back to simple vector store")
                self.collection = SimpleVectorStore()
                return

        # Set up the embedding function if possible
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            try:
                self.embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
                    model_name="all-MiniLM-L6-v2"
                )
                print("Created SentenceTransformer embedding function")
            except Exception as e:
                print(f"Error creating embedding function: {e}")
                self.embedding_function = None
        else:
            self.embedding_function = None

        # Create the collection
        try:
            if self.embedding_function:
                self.collection = self.client.create_collection(
                    name=collection_name,
                    embedding_function=self.embedding_function
                )
            else:
                self.collection = self.client.create_collection(name=collection_name)
            print(f"Created new collection: {collection_name}")
        except Exception as e:
            print(f"Error creating collection: {e}")
            print("Falling back to simple vector store")
            self.collection = SimpleVectorStore()

    def add_chunks(self, chunks: List[Dict[str, Any]]) -> None:
        if not chunks:
            print("Warning: No chunks to add to database")
            return

        # Process in smaller batches to avoid memory issues
        batch_size = 10
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i+batch_size]

            # Prepare data for batch addition
            ids = [chunk["chunk_id"] for chunk in batch]
            documents = [chunk["text"] for chunk in batch]
            metadatas = [chunk["metadata"] for chunk in batch]

            # Convert metadata to proper format (ensure all values are strings)
            for m in metadatas:
                for k, v in m.items():
                    if not isinstance(v, (str, int, float, bool)):
                        m[k] = str(v)

            try:
                # Add to collection
                self.collection.add(
                    ids=ids,
                    documents=documents,
                    metadatas=metadatas
                )
                print(f"Added batch of {len(batch)} chunks to vector database")
            except Exception as e:
                print(f"Error adding chunks to database: {e}")
                print("Skipping this batch")

        print(f"Added total of {len(chunks)} chunks to vector database")

    def query(self, query_text: str, n_results: int = 5, filters: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Query the vector database with optional metadata filters.
        """
        try:
            query_args = {
                "query_texts": [query_text],
                "n_results": min(n_results, 20)
            }

            if filters:
                query_args["where"] = filters  # <-- ChromaDB supports this

            results = self.collection.query(**query_args)
            return results
        except Exception as e:
            print(f"Error querying vector DB: {e}")
            return {
                "ids": [[]],
                "documents": [[]],
                "metadatas": [[]],
                "distances": [[]]
            }


from typing import List, Dict, Any, Optional, Tuple

# Add this class for table extraction
class TableExtractor:
    """Extracts tables from images using OCR and converts to structured JSON"""
    
    def __init__(self):
        self.confidence_threshold = 85  # OCR confidence threshold
    
    def extract_tables_from_image(self, image: Image.Image) -> List[Dict[str, Any]]:
        """Extract tables from an image and return as structured JSON"""
        # Use pytesseract with additional data options to find tables
        data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
        
        # Simple table detection based on line positions and text alignment
        tables = []
        current_table = None
        current_row = []
        last_line = -1
        
        # Process OCR results to identify tables
        for i in range(len(data['text'])):
            # Skip empty text or low confidence
            if not data['text'][i].strip() or int(data['conf'][i]) < self.confidence_threshold:
                continue
                
            # New line detection
            if data['line_num'][i] != last_line:
                if current_row and current_table is not None:
                    current_table['rows'].append(current_row)
                    current_row = []
                last_line = data['line_num'][i]
                
                # Check if this might be a new table header
                if self._is_likely_table_header(data['text'][i]):
                    if current_table is not None and current_table['rows']:
                        tables.append(current_table)
                    # Start a new table
                    current_table = {
                        'table_id': f"table_{uuid.uuid4()}",
                        'header': data['text'][i],
                        'rows': [],
                        'bbox': [data['left'][i], data['top'][i], 
                                data['width'][i], data['height'][i]]
                    }
            
            # Add to current row if we're in a table
            if current_table is not None:
                current_row.append(data['text'][i])
        
        # Add the last row and table if they exist
        if current_row and current_table is not None:
            current_table['rows'].append(current_row)
            tables.append(current_table)
            
        # Convert tables to more structured format
        structured_tables = []
        for table in tables:
            if len(table['rows']) > 1:  # Only include if there are actual rows
                structured = self._structure_table(table)
                if structured:
                    structured_tables.append(structured)
        
        return structured_tables
    
    def _is_likely_table_header(self, text: str) -> bool:
        """Check if text is likely to be a table header"""
        # Simple heuristic - headers often contain these words
        header_keywords = ['table', 'summary', 'total', 'year', 'quarter', 'month', 
                          'item', 'description', 'amount', 'value', 'date', 'name']
        
        text_lower = text.lower()
        # Check if text contains any header keywords
        return any(keyword in text_lower for keyword in header_keywords)
    
    def _structure_table(self, table: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Convert raw table data to structured format"""
        if not table['rows'] or len(table['rows']) < 2:
            return None
            
        # Try to determine columns based on first row
        if len(table['rows'][0]) > 1:
            headers = table['rows'][0]
        else:
            # If first row doesn't have multiple columns, use generic headers
            headers = [f"Column {i+1}" for i in range(max(len(row) for row in table['rows']))]
        
        # Process data rows
        data_rows = []
        for row in table['rows'][1:]:  # Skip header row
            if len(row) > 0:
                # Pad row if needed to match header length
                padded_row = row + [''] * (len(headers) - len(row))
                data_rows.append(padded_row[:len(headers)])  # Truncate if longer than headers
        
        # Create structured table
        return {
            'table_id': table['table_id'],
            'title': table['header'],
            'headers': headers,
            'data': data_rows
        }



class RagSystem:
    """Complete RAG system for PDF screenshots with table extraction and conversation memory"""

    def __init__(self,
                anthropic_api_key: str,
                model: str = "claude-3-7-sonnet-20250219"):
        self.pdf_processor = PDFScreenshotProcessor()
        self.chunker = ChunkStrategy(chunk_size=1000, chunk_overlap=200)
        self.vector_db = VectorDatabaseManager(collection_name="pdf_screenshots")
        self.tables_by_page = {}  # Store extracted tables
        self.pdf_path = None  # Store the current PDF path
        
        # Add conversation history
        self.conversation_history = []
        
        # Initialize Anthropic client
        self.anthropic = Anthropic(api_key=anthropic_api_key)
        self.model = model

        # Enhanced system prompt for Claude
        self.system_prompt = """
        You are an AI assistant specializing in answering questions about documents.

        You will receive context information extracted from PDF documents and possibly structured table data, along with a question and any previous conversation history.

        Your task is to:
        keep the answers human designed and not like a machine
        1. Answer based ONLY on the information provided in the context
        2. Answers should be human readable and should look like they were written by a human
        3.keep the answer short and concise
        4. If the context doesn't contain enough information, acknowledge the limitations
        5. Cite specific pages when referencing information (e.g., "According to page 3...")
        6. When referencing tables, cite the table title and page number
        7. Keep your responses clear, concise, and directly relevant to the question
        8. If there are OCR errors in the context, try to infer the correct meaning
        9.explain me like as if i am a 5 year old
        10.dont't give the same text from the context in your answer, rather make the meaning of it and then answer
        11. Include a confidence score (0-100%) at the end of your response based on how well the provided context answers the question
        12. Format your answer with clear paragraph breaks
        13. Consider previous conversation when relevant (e.g., if the user asks follow-up questions)
        
        Remember to cite your sources clearly using page numbers in the format: [Page X]
        """

    def index_pdf(self, pdf_path: str) -> None:
        self.pdf_path = pdf_path  # Store PDF path for later reference
        pages = self.pdf_processor.process_pdf(pdf_path)
        
        # Extract tables from pages
        self.tables_by_page = self.pdf_processor.extract_tables_from_pages(pages)
        
        # Save tables to disk for reference
        tables_json_path = f"{os.path.splitext(pdf_path)[0]}_tables.json"
        with open(tables_json_path, 'w') as f:
            json.dump(self.tables_by_page, f, indent=2)
        
        print(f"Extracted tables saved to {tables_json_path}")

        # Chunk the text from each page
        all_chunks = []
        for page in pages:
            # Prepare metadata
            metadata = {
                "pdf_name": page["pdf_name"],
                "page_num": page["page_num"],
                "page_id": page["page_id"],
                "has_tables": str(page["page_num"]) in self.tables_by_page
            }

            # Chunk the text
            page_chunks = self.chunker.chunk_text(page["text"], metadata)
            all_chunks.extend(page_chunks)

        # Add chunks to vector database
        self.vector_db.add_chunks(all_chunks)

        print(f"Indexed PDF: {pdf_path}")
        
        # Reset conversation history when indexing a new PDF
        self.conversation_history = []

    def query(self, query_text: str, n_results: int = 5) -> str:# Optional filters
        filters = {}
        if "page" in query_text.lower():
            match = re.search(r'page\s+(\d+)', query_text.lower())
            if match:
                filters["page_num"] = int(match.group(1))

        results = self.vector_db.query(query_text, n_results=n_results, filters=filters)
        retrieved = []
        for i in range(len(results["ids"][0])):
            chunk_text = results["documents"][0][i]
            metadata = results["metadatas"][0][i]
            retrieved.append({
        "text": chunk_text,
        "metadata": metadata
    })
        ranking_prompt = "Rank the following chunks from most to least relevant to the question:\n"
        ranking_prompt += f"\nQuestion: {query_text}\n\n"
        for i, item in enumerate(retrieved):
            ranking_prompt += f"Chunk {i+1}:\n{item['text'][:1000]}\n\n"

        ranking_prompt += "Return the order of chunk numbers from most to least relevant as a Python list (e.g., [3, 1, 2])"
        try:
            ranking_response = self.anthropic.messages.create(
        model=self.model,
        system="You are an expert ranking assistant.",
        messages=[{"role": "user", "content": ranking_prompt}],
        max_tokens=100,
        temperature=0.2
    )
            import ast
            import re

            try:
                raw_text = ranking_response.content[0].text.strip()
                print("Raw reranking response from Claude:\n", raw_text)

                 # Extract first list found in response using regex
                match = re.search(r"\[([0-9,\s]+)\]", raw_text)
                if match:
                    list_str = "[" + match.group(1) + "]"
                    rank_order = ast.literal_eval(list_str)
                else:
                    raise ValueError("No valid list found in Claude response.")

                print("Rank order:", rank_order)
            except Exception as e:
                print(f"error in reranking: {e}")
                rank_order = list(range(1, len(retrieved)+1))  # fallback to original order

            print('Rank order:', rank_order)
        except Exception as e: 
            print(f"error in reranking: {e}")
            rank_order = list(range(len(retrieved)))  # fallback to original order

# Reorder based on reranking
        reranked_chunks = [retrieved[i-1] for i in rank_order if 0 <= i-1 < len(retrieved)]
        
        

        # Check if results are empty
        if not results["ids"][0]:
            response = "Sorry, I couldn't find any relevant information in the document to answer your question."
            # Store in conversation history even if no results
            self.conversation_history.append({
                "question": query_text,
                "answer": response
            })
            return response

        # Prepare context for Claude
        context_parts = []
        page_nums = set()
        
        for i in range(len(results["ids"][0])):
            document = results["documents"][0][i]
            metadata = results["metadatas"][0][i]
            page_num = metadata.get('page_num', 'unknown')
            
            # Add page number to set
            if page_num != 'unknown':
                page_nums.add(str(page_num))
            
            context_part = f"--- EXCERPT FROM {metadata.get('pdf_name', 'document')}, PAGE {page_num} ---\n"
            context_part += document
            context_part += "\n---\n"
            
            context_parts.append(context_part)

        # Add relevant tables if they exist
        table_context = ""
        for page_num in page_nums:
            if page_num in self.tables_by_page:
                tables = self.tables_by_page[page_num]
                for table in tables:
                    table_context += f"\n--- TABLE FROM PAGE {page_num}: {table.get('title', 'Untitled Table')} ---\n"
                    
                    # Add headers
                    if 'headers' in table:
                        table_context += " | ".join(table['headers']) + "\n"
                        table_context += "-" * (sum(len(h) for h in table['headers']) + (len(table['headers'])-1) * 3) + "\n"
                    
                    # Add data rows
                    if 'data' in table:
                        for row in table['data']:
                            table_context += " | ".join(row) + "\n"
                    
                    table_context += "---\n"

        context = "\n".join(context_parts)
        if table_context:
            context += "\n\nTABLE DATA:\n" + table_context

        # Add conversation history to provide context for follow-up questions
        conversation_context = ""
        if self.conversation_history:
            conversation_context = "\n\nPREVIOUS CONVERSATION:\n"
            for i, exchange in enumerate(self.conversation_history[-3:]):  # Include up to last 3 exchanges
                conversation_context += f"Question {i+1}: {exchange['question']}\n"
                conversation_context += f"Answer {i+1}: {exchange['answer']}\n\n"

        # Create prompt for Claude
        user_message = f"""
        CONTEXT INFORMATION:
        {context}
        
        {conversation_context}

        CURRENT QUESTION:
        {query_text}

        Please answer the question based on the provided context information.
        Format your answer with clear citations to page numbers in [Page X] format.
        If tables are relevant to the answer, refer to them specifically.
        If this appears to be a follow-up to a previous question, take that previous into account.
        End your response with a confidence score (0-100%) that reflects how well the context answers the question.
        """

        try:
            # Generate response
            response = self.anthropic.messages.create(
                model=self.model,
                system=self.system_prompt,
                messages=[
                    {"role": "user", "content": user_message}
                ],
                max_tokens=1500,
                temperature=0.2
            )
            
            answer = response.content[0].text
            
            # Add page review links
            if self.pdf_path:
                pdf_filename = os.path.basename(self.pdf_path)
                answer += f"\n\nSource document: {pdf_filename}"
                
                # Extract page numbers from response using regex
                page_citations = re.findall(r'\[Page (\d+)\]', answer)
                if page_citations:
                    answer += "\n\nRelevant pages:"
                    for page in sorted(set(page_citations)):
                        answer += f"\n- Page {page}"
            
            # Store in conversation history
            self.conversation_history.append({
                "question": query_text,
                "answer": answer
            })
            
            return answer
        except Exception as e:
            print(f"Error generating response: {e}")
            error_msg = f"Sorry, I encountered an error while generating a response: {str(e)}"
            
            # Store error in conversation history
            self.conversation_history.append({
                "question": query_text,
                "answer": error_msg
            })
            
            return error_msg

    def save_conversation(self, output_file: str = None) -> str:
        """Save the conversation history to a file"""
        if not output_file:
            pdf_name = os.path.splitext(os.path.basename(self.pdf_path))[0] if self.pdf_path else "conversation"
            output_file = f"{pdf_name}_conversation.json"
        
        with open(output_file, 'w') as f:
            json.dump(self.conversation_history, f, indent=2)
        
        print(f"Conversation history saved to {output_file}")
        return output_file

    def load_conversation(self, input_file: str) -> bool:
        """Load conversation history from a file"""
        try:
            with open(input_file, 'r') as f:
                self.conversation_history = json.load(f)
            print(f"Loaded {len(self.conversation_history)} conversation exchanges from {input_file}")
            return True
        except Exception as e:
            print(f"Error loading conversation history: {e}")
            return False
if __name__ == "__main__":
    # Print import status only when running directly
    if PDF2IMAGE_AVAILABLE:
        print("Successfully imported pdf2image")
    else:
        print("Error importing pdf2image")
        print("Will use PyMuPDF as fallback")

    if PYMUPDF_AVAILABLE:
        print("Successfully imported PyMuPDF")
    else:
        print("Error importing PyMuPDF")
        print("Try running: pip install pymupdf")

    if CHROMADB_AVAILABLE:
        print("Successfully imported ChromaDB")
    else:
        print("Error importing ChromaDB")
        print("Will use simple vector store fallback")

    if SENTENCE_TRANSFORMERS_AVAILABLE:
        print("Successfully imported SentenceTransformer")
    else:
        print("Error importing SentenceTransformer")

    # Test code can go here
    print("Tesseract command set to:", pytesseract.pytesseract.tesseract_cmd)


