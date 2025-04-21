from flask import Flask, request, jsonify
from flask_cors import CORS
from rag import RagSystem
import shutil
from pathlib import Path
import pytesseract
import datetime
import json

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

app = Flask(__name__)
CORS(app)

def initialize_rag_system():
    print("\nInitializing RAG system...")
    anthropic_api_key = "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"
    pdf_path = "ltimindtree_annual_report.pdf"
    
    global rag_system
    rag_system = RagSystem(
        anthropic_api_key=anthropic_api_key,
        model="claude-3-7-sonnet-20250219"
    )
    rag_system.index_pdf(pdf_path)
    print("Indexing complete! The system is ready for queries.\n")

initialize_rag_system()

@app.route('/history', methods=['GET'])
def list_conversation():
    if not rag_system.conversation_history:
        return jsonify({"message": "No conversation history yet."})
    
    return jsonify({"history": rag_system.conversation_history})

@app.route('/save', methods=['POST'])
def save_conversation():
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"conversation_{timestamp}.json"
    
    with open(output_file, 'w') as f:
        json.dump(rag_system.conversation_history, f, indent=2)
    
    return jsonify({"message": f"Conversation saved to {output_file}"})

@app.route('/clear', methods=['POST'])
def clear_history():
    rag_system.conversation_history = []
    return jsonify({"message": "Conversation history cleared."})


@app.route('/retell', methods=['POST'])
def retell():
    data = request.get_json()
    
    question = data['args']['question']
    print(question)
    response = rag_system.query(question)
    print(response)
    print("\n" + "-"*80 + "\n")
    return jsonify({"answer": response})

@app.route('/query', methods=['POST'])
def query():
    data = request.get_json()  
    print("Raw data received:", data)
    
    question = data['question']
    print("✅ Question received:", question)
    response = rag_system.query(question)
    print("\nAnswer:")
    print(response)
    print("\n" + "-"*80 + "\n")
    return jsonify({"answer": response})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)