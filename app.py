# app.py (HUGGINGFACE / GRADIO ready)
import os
import sys
import torch
torch.set_num_threads(1)
torch.set_num_interop_threads(1)

import gradio as gr
from PyPDF2 import PdfReader
import textstat
import time

# Import your summarization function from your local file summarize.py
# Make sure the file is named summarize.py and contains summarize_text(...)
from summarize import summarize_text

# small safety limits
MAX_CHARS = 3000  # truncate very long text
length_map = {
    "Short": (30, 80),
    "Medium": (80, 150),
    "Detailed": (120, 200)
}

# history stored in memory (session will reset between restarts)
summary_history = []

# helper to read uploaded file
def read_file(file_obj):
    if file_obj is None:
        return ""
    name = getattr(file_obj, "name", "")
    # file_obj may be BytesIO; Hugging Face passes a tempfile path sometimes
    try:
        # If it's a file-like object
        if hasattr(file_obj, "read"):
            data = file_obj.read()
            # if bytes decode
            if isinstance(data, (bytes, bytearray)):
                try:
                    return data.decode("utf-8")
                except:
                    # try pdf
                    pass
    except Exception:
        pass

    # fallback: if file_obj is a path string
    try:
        # If HF passes a local path
        if isinstance(file_obj, str) and os.path.exists(file_obj):
            if file_obj.lower().endswith(".pdf"):
                reader = PdfReader(file_obj)
                text = "\n".join([p.extract_text() or "" for p in reader.pages])
                return text
            else:
                with open(file_obj, "r", encoding="utf-8", errors="ignore") as f:
                    return f.read()
    except Exception:
        pass

    # try treating file_obj as upload with .name and content
    try:
        if hasattr(file_obj, "name") and file_obj.name.lower().endswith(".pdf"):
            reader = PdfReader(file_obj)
            text = "\n".join([p.extract_text() or "" for p in reader.pages])
            return text
    except Exception:
        pass

    return ""

# summarization wrapper used by Gradio
def summarize_with_status(*args):
    """
    Accept arbitrary args from Gradio. We expect at least:
    args[0] = text_input (str)
    args[1] = file_input (may be None)
    args[2] = summary_length (str)  OR some ordering depending on your UI.
    To be robust we try to detect types.
    """
    # default values
    text_input = ""
    file_input = None
    summary_length = "Medium"

    # map incoming args flexibly
    for a in args:
        if isinstance(a, str):
            # choose the first long-ish string as text input or a length option
            if a in length_map.keys():
                summary_length = a
            elif len(a) < 30 and a in length_map.keys():
                summary_length = a
            elif not text_input:
                text_input = a
        elif hasattr(a, "name") or isinstance(a, (bytes, bytearray)):
            file_input = a
        elif a is None:
            continue
        else:
            # fallback: if Arg is gr.File represented as dict, try to extract "name"
            try:
                if isinstance(a, dict) and "name" in a:
                    file_input = a
            except:
                pass

    # if file provided, read it (PDF or txt)
    if file_input:
        file_text = read_file(file_input)
        if file_text:
            text_input = file_text

    if not text_input or not text_input.strip():
        return "⚠️ Please enter text or upload a valid .txt/.pdf file.", "", ""

    # truncate to avoid OOM on Spaces free tier
    warning = ""
    if len(text_input) > MAX_CHARS:
        text_input = text_input[:MAX_CHARS]
        warning = f"⚠️ Text truncated to {MAX_CHARS} characters to prevent memory issues.\n\n"

    # chosen lengths
    min_len, max_len = length_map.get(summary_length, length_map["Medium"])

    start_time = time.time()
    try:
        # call your summarize_text from summarize.py
        # ensure your summarize_text signature supports (text, max_length=?, min_length=?)
        summary = summarize_text(text_input, max_length=max_len, min_length=min_len)
    except Exception as e:
        # print stacktrace to logs (HF shows these)
        import traceback
        traceback.print_exc()
        return "❌ Summarization failed. Check logs.", "", ""

    end_time = time.time()

    # stats
    original_len = len(text_input.split())
    summary_len = len(summary.split())
    reduction = round(100 - (summary_len / original_len * 100), 2) if original_len else 0
    try:
        readability = textstat.flesch_reading_ease(summary)
    except Exception:
        readability = 0.0
    processing_time = round(end_time - start_time, 2)

    stats = (
        f"📄 Words: {summary_len} | 📉 Reduced: {reduction}%\n"
        f"⏱ Time: {processing_time}s | 📚 Readability: {readability:.2f}"
    )

    # save history
    summary_history.append({
        "summary": summary,
        "length": summary_length,
        "original_len": original_len,
        "summary_len": summary_len,
        "reduction": reduction,
        "time": processing_time,
        "readability": readability
    })

    # prepare comparison text if exists
    comparison_text = ""
    if len(summary_history) > 1:
        comparison_text = (
            "### 🔍 Compare with Previous Summary\n\n"
            f"#### 🕒 Current Summary\n\n{summary_history[-1]['summary']}\n\n"
            f"#### 🕔 Previous Summary\n\n{summary_history[-2]['summary']}"
        )

    return warning + summary, stats, comparison_text

def clear_history():
    summary_history.clear()
    return "", "", ""

# ------------------ Gradio Interface ------------------
with gr.Blocks() as demo:
    gr.Markdown(
        """
        <h1 style='text-align:center; color:#0072B2;'>🧠 TextSummarizer Pro</h1>
        <p style='text-align:center;'>Transform long text or PDFs into concise summaries instantly!</p>
        <hr>
        """
    )

    with gr.Row():
        with gr.Column(scale=1):
            input_method = gr.Radio(
                ["Enter Text", "Upload File"],
                label="📥 Choose Input Method",
                value="Enter Text"
            )

            text_input = gr.Textbox(label="📝 Enter Text", lines=10, visible=True)
            file_input = gr.File(label="📄 Upload .txt or .pdf", file_count="single", visible=False)

            def toggle_inputs(method):
                if method == "Enter Text":
                    return gr.update(visible=True), gr.update(visible=False)
                else:
                    return gr.update(visible=False), gr.update(visible=True)

            input_method.change(
                fn=toggle_inputs,
                inputs=input_method,
                outputs=[text_input, file_input]
            )

        with gr.Column(scale=1):
            summary_length = gr.Radio(["Short", "Medium", "Detailed"], label="🧩 Choose Summary Length", value="Medium")
            summarize_button = gr.Button("✨ Generate Summary")
            clear_button = gr.Button("🗑️ Clear Summary History")

    output_summary = gr.Markdown(label="✅ Summary Output")
    output_stats = gr.Textbox(label="📊 Summary Stats")
    output_comparison = gr.Markdown(label="🔍 Summary Comparison")

    # connect: pass text_input, file_input, summary_length to the function
    summarize_button.click(
        fn=summarize_with_status,
        inputs=[text_input, file_input, summary_length],
        outputs=[output_summary, output_stats, output_comparison]
    )

    clear_button.click(
        fn=clear_history,
        inputs=None,
        outputs=[output_summary, output_stats, output_comparison]
    )

    gr.Markdown(
        """
        ---
        <p style='text-align:center;'>💡 Built by <b>Panduru Aadhithya</b> | B.Tech AI & DS</p>
        <p style='text-align:center;'>Powered by Hugging Face 🤗 + Transformers + PyTorch</p>
        """
    )

# Use server_name so HF runs the app correctly
demo.launch(server_name="0.0.0.0")# app.py - TextSummarizer Pro (all-features best-effort)
import os
import sys
import time
import tempfile
import json
import string
import collections
import traceback

# Reduce threads for stability on Spaces / small hosts
import torch
torch.set_num_threads(1)
torch.set_num_interop_threads(1)

import gradio as gr
from transformers import pipeline
from PyPDF2 import PdfReader
import textstat

# Optional imports - enable features if available
HAS_DOCX = False
HAS_REPORTLAB = False
HAS_GTTS = False
HAS_NLTK = False
HAS_LANGDETECT = False

try:
    import docx  # python-docx
    HAS_DOCX = True
except Exception:
    HAS_DOCX = False

try:
    from reportlab.lib.pagesizes import A4
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
    from reportlab.lib.styles import getSampleStyleSheet
    HAS_REPORTLAB = True
except Exception:
    HAS_REPORTLAB = False

try:
    from gtts import gTTS
    HAS_GTTS = True
except Exception:
    HAS_GTTS = False

try:
    import nltk
    nltk.data.find("tokenizers/punkt")
    HAS_NLTK = True
except Exception:
    HAS_NLTK = False
    try:
        # attempt download (may not be allowed on HF; wrap in try)
        nltk.download("punkt")
        nltk.download("stopwords")
        HAS_NLTK = True
    except Exception:
        HAS_NLTK = False

try:
    from langdetect import detect
    HAS_LANGDETECT = True
except Exception:
    HAS_LANGDETECT = False

# ---------------- Configuration ----------------
MODEL_NAME = os.environ.get("SUMMARIZER_MODEL", "t5-small")  # small & stable
CHUNK_CHAR_SIZE = 1400
MAX_INPUT_CHARS = 25000
TEMP_DIR = tempfile.gettempdir()
HISTORY_LIMIT = 12

# Load model pipeline once (safe for CPU)
print("Loading summarization model:", MODEL_NAME)
try:
    summarizer = pipeline("summarization", model=MODEL_NAME, truncation=True, device=-1)
    print("Model loaded OK")
except Exception as e:
    print("Model load failed:", e)
    traceback.print_exc()
    raise

# ---------------- Helpers ----------------
def read_file(file_obj):
    """Read file input (path, file-like, HF dict). Supports .pdf, .txt, .docx (if python-docx installed)."""
    if not file_obj:
        return ""
    # HF sometimes passes dict with "name" and "data"
    try:
        if isinstance(file_obj, dict) and "name" in file_obj and "data" in file_obj:
            # try tempfile path in dict
            path = file_obj.get("tempfile") or file_obj.get("name")
            if path and os.path.exists(path):
                return read_file(path)
    except Exception:
        pass

    # if path string
    if isinstance(file_obj, str):
        path = file_obj
        if os.path.exists(path):
            if path.lower().endswith(".pdf"):
                try:
                    reader = PdfReader(path)
                    pages = [p.extract_text() or "" for p in reader.pages]
                    return "\n".join(pages)
                except Exception:
                    return ""
            elif path.lower().endswith(".docx") and HAS_DOCX:
                try:
                    doc = docx.Document(path)
                    return "\n".join([p.text for p in doc.paragraphs])
                except Exception:
                    return ""
            else:
                try:
                    with open(path, "r", encoding="utf-8", errors="ignore") as f:
                        return f.read()
                except Exception:
                    return ""
        else:
            return ""

    # file-like object (Uploaded)
    try:
        if hasattr(file_obj, "read"):
            data = file_obj.read()
            # bytes -> try decode as text
            if isinstance(data, (bytes, bytearray)):
                try:
                    text = data.decode("utf-8")
                    return text
                except Exception:
                    # try treat as pdf bytes
                    try:
                        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
                        tmp.write(data)
                        tmp.close()
                        txt = read_file(tmp.name)
                        os.unlink(tmp.name)
                        return txt
                    except Exception:
                        return ""
            elif isinstance(data, str):
                return data
    except Exception:
        pass

    return ""

def chunk_text_map_reduce(text, chunk_chars=CHUNK_CHAR_SIZE):
    """Map-reduce chunking summarizer (safe for long inputs)."""
    t = text.strip()
    if not t:
        return ""
    if len(t) > MAX_INPUT_CHARS:
        t = t[:MAX_INPUT_CHARS]

    # split into chunks trying to respect sentence boundaries
    chunks = []
    i = 0
    L = len(t)
    while i < L:
        end = min(i + chunk_chars, L)
        # try to expand to a dot within a small window to avoid mid-sentence cuts
        if end < L:
            look = t.rfind(".", i, min(L, end + 200))
            if look > i:
                end = look + 1
        chunk = t[i:end].strip()
        if chunk:
            chunks.append(chunk)
        i = end

    # map: summarize each chunk
    partial_summaries = []
    for c in chunks:
        try:
            out = summarizer(c, max_length=150, min_length=30, do_sample=False)
            s = out[0].get("summary_text", "")
            partial_summaries.append(s.strip())
        except Exception:
            # fallback: truncate chunk
            partial_summaries.append(c[:500])

    # reduce: summarize combined partials
    if not partial_summaries:
        return ""
    if len(partial_summaries) == 1:
        return partial_summaries[0]
    combined = " ".join(partial_summaries)
    try:
        final = summarizer(combined, max_length=200, min_length=40, do_sample=False)
        return final[0].get("summary_text", combined)
    except Exception:
        return combined

def extract_keywords(text, top_k=8):
    """Simple frequency-based keywords (lightweight)."""
    s = text.lower()
    table = str.maketrans("", "", string.punctuation)
    words = s.translate(table).split()
    stopwords = {
        "the","and","is","in","it","of","to","a","that","he","she","they","was","were","on","for","with","as","his","her",
        "at","by","an","be","this","which","or","from","but","not","are","have","had","has","their","its","you","i"
    }
    freq = collections.Counter(w for w in words if w not in stopwords and len(w) > 2)
    return [k for k,_ in freq.most_common(top_k)]

def generate_title(text):
    kws = extract_keywords(text, top_k=5)
    if kws:
        return " ".join(w.capitalize() for w in kws[:5])
    # fallback first sentence
    s = text.strip().split(".")
    return s[0][:60] if s else "Summary"

def write_pdf(summary_text, title="Summary"):
    """Return path to PDF file if reportlab available, else None."""
    if not HAS_REPORTLAB:
        return None
    try:
        fname = f"summary_{int(time.time())}.pdf"
        path = os.path.join(TEMP_DIR, fname)
        doc = SimpleDocTemplate(path, pagesize=A4)
        styles = getSampleStyleSheet()
        story = []
        story.append(Paragraph(f"<b>{title}</b>", styles["Title"]))
        story.append(Spacer(1, 8))
        for line in summary_text.split("\n"):
            story.append(Paragraph(line.replace("\n", "<br/>"), styles["BodyText"]))
            story.append(Spacer(1, 4))
        doc.build(story)
        return path
    except Exception:
        traceback.print_exc()
        return None

def make_audio(summary_text, lang="en"):
    """Generate mp3 audio using gTTS if available; returns path or None."""
    if not HAS_GTTS:
        return None
    try:
        tts = gTTS(text=summary_text, lang=lang)
        fname = f"summary_{int(time.time())}.mp3"
        path = os.path.join(TEMP_DIR, fname)
        tts.save(path)
        return path
    except Exception:
        traceback.print_exc()
        return None

def summary_stats(original_text, summary_text, elapsed):
    orig_words = len(original_text.split())
    sum_words = len(summary_text.split())
    reduction = round(100 - (sum_words / orig_words * 100), 2) if orig_words else 0.0
    try:
        readability = textstat.flesch_reading_ease(summary_text)
    except Exception:
        readability = 0.0
    stats = f"Original words: {orig_words} | Summary words: {sum_words} | Reduced: {reduction}%\nTime: {round(elapsed,2)}s | Readability: {readability:.2f}"
    return stats

# ---------------- History ----------------
HISTORY = []

# ---------------- Main processing function ----------------
def process_input(text_input, file_input, length_choice, style_choice, do_title, do_keywords, do_pdf, do_audio):
    """
    inputs:
      - text_input: str
      - file_input: file-like / path
      - length_choice: 'Short'/'Medium'/'Detailed'
      - style_choice: 'Simple'/'Bulleted'/'Executive'
      - do_title, do_keywords, do_pdf, do_audio: booleans
    returns: summary_markdown, stats_text, keywords_json, comparison_md, download_txt, download_pdf, download_audio
    """
    # get text from file if provided
    text = ""
    if file_input:
        text = read_file(file_input)
    if not text and text_input:
        text = text_input

    if not text or not text.strip():
        return "⚠️ Please paste text or upload a .txt/.pdf/.docx file.", "", "[]", "", None, None, None

    if len(text) > MAX_INPUT_CHARS:
        text = text[:MAX_INPUT_CHARS]
        truncated_note = f"⚠️ Input truncated to {MAX_INPUT_CHARS} characters for stability.\n\n"
    else:
        truncated_note = ""

    # adjust summarizer target lengths (we call map-reduce which itself uses set lengths)
    # We will apply final condense for Short
    start = time.time()
    try:
        raw_summary = chunk_text_map_reduce(text, chunk_chars=CHUNK_CHAR_SIZE)
        if length_choice == "Short":
            try:
                # condense raw_summary further
                raw_summary = summarizer(raw_summary, max_length=80, min_length=20, do_sample=False)[0].get("summary_text", raw_summary)
            except Exception:
                pass
    except Exception:
        traceback.print_exc()
        return "❌ Summarization failed (internal). Check logs.", "", "[]", "", None, None, None
    elapsed = time.time() - start

    # style adjustments
    final_summary = raw_summary
    if style_choice == "Bulleted":
        # split into sentences (naive) and present as bullets
        if HAS_NLTK:
            try:
                from nltk.tokenize import sent_tokenize
                sents = sent_tokenize(final_summary)
            except Exception:
                sents = [s.strip() for s in final_summary.split(". ") if s.strip()]
        else:
            sents = [s.strip() for s in final_summary.split(". ") if s.strip()]
        bullets = "\n".join(["- " + s.strip().rstrip(".") for s in sents[:8]])
        final_summary = bullets
    elif style_choice == "Executive":
        # keep as-is (longer)
        pass
    else:
        # Simple: keep as-is (could also shorten slightly)
        pass

    # title & keywords
    title = generate_title(text) if do_title else ""
    keywords = extract_keywords(text, top_k=10) if do_keywords else []

    # prepare stats & history
    stats = summary_stats(text, final_summary, elapsed)

    entry = {
        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "title": title,
        "summary": final_summary,
        "keywords": keywords,
        "stats": stats
    }
    HISTORY.append(entry)
    if len(HISTORY) > HISTORY_LIMIT:
        HISTORY.pop(0)

    # output comparison (previous vs latest)
    comparison = ""
    if len(HISTORY) >= 2:
        prev = HISTORY[-2]
        cur = HISTORY[-1]
        comparison = f"### 🔍 Compare\n\n#### Latest ({cur['time']})\n{cur['summary']}\n\n#### Previous ({prev['time']})\n{prev['summary']}"

    # prepare download files: txt always
    txt_path = None
    try:
        fname = f"summary_{int(time.time())}.txt"
        txt_path = os.path.join(TEMP_DIR, fname)
        with open(txt_path, "w", encoding="utf-8") as f:
            if title:
                f.write(f"Title: {title}\n\n")
            f.write(final_summary)
    except Exception:
        txt_path = None

    pdf_path = None
    if do_pdf and HAS_REPORTLAB:
        pdf_path = write_pdf(final_summary, title=title if title else "Summary")

    audio_path = None
    if do_audio and HAS_GTTS:
        # choose language via langdetect if available
        lang = "en"
        if HAS_LANGDETECT:
            try:
                detected = detect(final_summary[:200])
                lang = detected if detected else "en"
            except Exception:
                lang = "en"
        audio_path = make_audio(final_summary, lang=lang)

    summary_md = truncated_note + (f"**Title:** {title}\n\n" if title else "") + final_summary

    return summary_md, stats, json.dumps(keywords), comparison, (txt_path if txt_path else None), (pdf_path if pdf_path else None), (audio_path if audio_path else None)

def clear_history():
    HISTORY.clear()
    return "", "", "", None, None, None, None

# ---------------- Gradio UI ----------------
with gr.Blocks(css="""
    .hero {text-align:center}
    .summary-box {background:#f7f9fc;padding:12px;border-radius:8px}
    .small {font-size:0.9rem;color:#555}
""") as demo:
    gr.Markdown("<div class='hero'><h1>🧠 TextSummarizer Pro — All Features</h1><p>Summarize long text, PDFs and DOCX; download as TXT/PDF/Audio; extract keywords & title.</p></div>")

    with gr.Row():
        with gr.Column(scale=1):
            input_method = gr.Radio(["Enter Text", "Upload File"], value="Enter Text", label="📥 Choose Input Method")
            text_input = gr.Textbox(label="📝 Enter Text", lines=12, placeholder="Paste your story or article here...")
            file_input = gr.File(label="📄 Upload .txt, .pdf or .docx", file_count="single", visible=False)

            def toggle(choice):
                if choice == "Enter Text":
                    return gr.update(visible=True), gr.update(visible=False)
                else:
                    return gr.update(visible=False), gr.update(visible=True)

            input_method.change(toggle, inputs=input_method, outputs=[text_input, file_input])

        with gr.Column(scale=1):
            length_choice = gr.Radio(["Short", "Medium", "Detailed"], value="Medium", label="🧩 Summary Length")
            style_choice = gr.Radio(["Simple", "Bulleted", "Executive"], value="Simple", label="✳️ Summary Style")
            do_title = gr.Checkbox(label="Generate Title", value=True)
            do_keywords = gr.Checkbox(label="Extract Keywords", value=True)
            do_pdf = gr.Checkbox(label=f"Export PDF (reportlab {'available' if HAS_REPORTLAB else 'missing'})", value=False, visible=HAS_REPORTLAB)
            do_audio = gr.Checkbox(label=f"Generate Audio (gTTS {'available' if HAS_GTTS else 'missing'})", value=False, visible=HAS_GTTS)
            summarize_btn = gr.Button("✨ Generate Summary")
            clear_btn = gr.Button("🗑️ Clear History")

    with gr.Row():
        with gr.Column(scale=2):
            output_summary = gr.Markdown(label="✅ Summary Output")
            with gr.Row():
                download_txt = gr.File(label="⬇️ Download Summary (.txt)")
                download_pdf = gr.File(label="⬇️ Download Summary (.pdf)", visible=HAS_REPORTLAB)
                download_audio = gr.File(label="⬇️ Download Audio (.mp3)", visible=HAS_GTTS)

        with gr.Column(scale=1):
            output_stats = gr.Textbox(label="📊 Summary Stats")
            output_keywords = gr.Textbox(label="🔑 Keywords (JSON)")
            output_comparison = gr.Markdown(label="🔍 Summary Comparison")

    # wiring
    summarize_btn.click(
        fn=process_input,
        inputs=[text_input, file_input, length_choice, style_choice, do_title, do_keywords, do_pdf, do_audio],
        outputs=[output_summary, output_stats, output_keywords, output_comparison, download_txt, download_pdf, download_audio]
    )

    clear_btn.click(
        fn=clear_history,
        inputs=None,
        outputs=[output_summary, output_stats, output_keywords, download_txt, download_pdf, download_audio]
    )

    gr.Markdown("<hr><p style='text-align:center;'>Built by <b>Panduru Aadhithya</b> — Powered by Gradio & Transformers</p>")

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0")

