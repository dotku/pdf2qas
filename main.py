import os
from dotenv import load_dotenv
import fitz  # PyMuPDF for PDF text extraction
import csv
import openai
import random

# OpenAI API key
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
openai.api_key = api_key
print('api_key', api_key)

# Function to extract text from PDF
def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text("text") + "\n"
    return text

# Function to split text into chunks
def split_text(text, chunk_size=500):
    words = text.split()
    chunks = [" ".join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]
    return chunks

# Function to generate a question and answer based on context
def generate_qa(context):
    prompt = f"Based on the following context, generate a question and an answer:\n\n{context}\n\nQuestion: "
    
    response = openai.ChatCompletion.create(
        model="gpt-4",  # or "gpt-3.5-turbo"
        messages=[
            {
                "role": "system",
                "content": "你是一个乐于助人的 AI，会生成问答对。请按照如下格式返回输出：\n问题: <你的问题>\n回答: <你的回答>"
            },
            {"role": "user", "content": prompt}
        ],
        temperature=0.7
    )
    
    qa_text = response["choices"][0]["message"]["content"]
    
    # Extract question and answer from response
    qa_lines = qa_text.split("\n")
    question = qa_lines[0].replace("问题: ", "").strip() if len(qa_lines) > 0 else "未生成问题"
    answer = qa_lines[1].replace("回答: ", "").strip() if len(qa_lines) > 1 else "未生成回答"
    
    return question, answer

# Main function
def process_pdf_to_csv(pdf_path, csv_filename, num_questions=10):
    extracted_text = extract_text_from_pdf(pdf_path)
    contexts = split_text(extracted_text, chunk_size=500)
    
    # Select 10 random contexts if more than 10 exist
    selected_contexts = random.sample(contexts, min(num_questions, len(contexts)))

    with open(csv_filename, mode="w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["Context", "Question", "Answer"])

        for context in selected_contexts:
            question, answer = generate_qa(context)
            writer.writerow([context, question, answer])

    print(f"CSV file '{csv_filename}' with {num_questions} Q&A pairs generated successfully!")

# Example Usage
pdf_file_path = "example.pdf"  # Replace with your PDF file path
csv_output_path = "qa_dataset.csv"
process_pdf_to_csv(pdf_file_path, csv_output_path, num_questions=10)
