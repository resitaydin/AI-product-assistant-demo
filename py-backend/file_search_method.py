from transformers import pipeline
import numpy as np

# Load and preprocess the file
def load_text_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()
    return text

file_path = 'moisturizer.txt'
file_content = load_text_file(file_path)

# Segment the text
def segment_text(text, max_length=1000):
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

# Load a larger or specialized pre-trained question-answering model and tokenizer
qa_pipeline = pipeline('question-answering', model='bert-large-uncased-whole-word-masking-finetuned-squad', tokenizer='bert-large-uncased')

def answer_question(question, context):
    return qa_pipeline(question=question, context=context)

# Function to find the most relevant segment
def retrieve_relevant_segment(question, segments):
    answers = []
    for segment in segments:
        result = answer_question(question, segment)
        answers.append((result['score'], result['answer'], segment))
    
    # Sort answers by confidence score
    answers = sorted(answers, key=lambda x: x[0], reverse=True)
    return answers[0] if answers else ("No answer found", "", "")

# Example usage
question = "What type of testing process does it go through?"
best_answer = retrieve_relevant_segment(question, segments)

print(f"Answer: {best_answer[1]}")
print(f"Confidence Score: {best_answer[0]}")
