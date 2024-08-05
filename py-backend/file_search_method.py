import torch
import numpy as np
from transformers import DistilBertTokenizer, DistilBertModel, pipeline
from sklearn.metrics.pairwise import cosine_similarity

# Load and preprocess the file
def load_text_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()
    return text

file_path = 'moisturizer.txt'
file_content = load_text_file(file_path)

# Segment the text
def segment_text(text, max_length=500):
    sentences = text.split('. ')
    segments = []
    current_segment = []
    current_length = 0

    for sentence in sentences:
        sentence_length = len(sentence.split())
        if current_length + sentence_length > max_length:
            segments.append(' '.join(current_segment))
            current_segment = []
            current_length = 0
        current_segment.append(sentence)
        current_length += sentence_length
    
    if current_segment:
        segments.append(' '.join(current_segment))

    return segments

segments = segment_text(file_content)

# Generate embeddings for each segment
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
model = DistilBertModel.from_pretrained('distilbert-base-uncased')

def get_embeddings(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).numpy()

segment_embeddings = [get_embeddings(segment) for segment in segments]
segment_embeddings = np.vstack(segment_embeddings)

# Retrieve relevant segment
def retrieve_relevant_segment(question, segment_embeddings, segments):
    question_embedding = get_embeddings(question)
    similarities = cosine_similarity(question_embedding, segment_embeddings)
    best_match_idx = similarities.argmax()
    return segments[best_match_idx]

# Answer questions using the LLM
qa_pipeline = pipeline('question-answering', model='distilbert-base-uncased')

def answer_question(question, context):
    return qa_pipeline(question=question, context=context)

# Example usage
question = "What is the main topic of the document?"
relevant_segment = retrieve_relevant_segment(question, segment_embeddings, segments)
answer = answer_question(question, relevant_segment)
print(f"Answer: {answer['answer']}")
