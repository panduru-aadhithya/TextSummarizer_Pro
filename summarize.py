from transformers import pipeline

summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

def chunk_text(text, chunk_size=3000):
    return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

def summarize_text(text, max_length=250, min_length=50):
    chunks = chunk_text(text)
    summaries = []

    for chunk in chunks:
        summary = summarizer(
            chunk,
            max_length=max_length,
            min_length=min_length,
            do_sample=False,
            truncation=True
        )[0]["summary_text"]
        summaries.append(summary)

    return " ".join(summaries)
