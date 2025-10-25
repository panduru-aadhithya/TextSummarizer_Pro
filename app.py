import streamlit as st
from summarize import summarize_text
import time
import textstat
from PyPDF2 import PdfReader
import io

# ----------------------- PAGE CONFIG -----------------------
st.set_page_config(
    page_title="TextSummarizer Pro",
    page_icon="🧠",
    layout="centered",
    initial_sidebar_state="expanded"
)

# >>> Added for memory optimization <<<
from transformers import pipeline

@st.cache_resource
def load_summarizer():
    """Load summarization model only once."""
    summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")
    return summarizer

summarizer_model = load_summarizer()
MAX_CHARS = 3000  # prevent very long text inputs
# >>> End memory optimization <<<


# ----------------------- SIDEBAR ----------------------------
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/4712/4712109.png", width=100)
    st.title("🧠 TextSummarizer Pro")
    st.markdown("**Version 4.0 — Memory + Comparison Mode**")
    st.markdown("---")
    st.markdown("### 👤 Created by:")
    st.markdown("**Panduru Aadhithya** — B.Tech AI & DS Student")
    st.markdown("💼 Internship Project | Streamlit + Hugging Face")
    st.markdown("---")
    theme_choice = st.radio("🎨 Choose Theme Mode", ["Light", "Dark"])
    st.markdown("---")
    st.markdown("💡 Summarize long articles, research papers, or PDFs — now with summary history!")

# ----------------------- TITLE -------------------------
if theme_choice == "Dark":
    st.markdown(
        "<h1 style='text-align:center; color:#00FFFF;'>🧠 TextSummarizer Pro</h1>",
        unsafe_allow_html=True
    )
else:
    st.markdown(
        "<h1 style='text-align:center; color:#0072B2;'>🧠 TextSummarizer Pro</h1>",
        unsafe_allow_html=True
    )
st.markdown("### Transform long text or documents into short, smart summaries instantly!")
st.markdown("---")

# ----------------------- SESSION STATE ----------------------
if "summaries" not in st.session_state:
    st.session_state.summaries = []  # store all summaries with metadata

# ----------------------- INPUT ----------------------
option = st.radio("📥 Choose Input Method", ["Enter Text", "Upload File"])
text_input = ""

if option == "Enter Text":
    text_input = st.text_area("📝 Enter your text below:", height=200)
elif option == "Upload File":
    uploaded_file = st.file_uploader("📄 Upload a .txt or .pdf file", type=["txt", "pdf"])
    if uploaded_file:
        if uploaded_file.type == "application/pdf":
            reader = PdfReader(uploaded_file)
            text_input = "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])
        else:
            text_input = uploaded_file.read().decode("utf-8")

# >>> Added for memory optimization <<<
# Truncate text if it's too long to prevent memory overload
if len(text_input) > MAX_CHARS:
    text_input = text_input[:MAX_CHARS]
    st.warning(f"⚠️ Text truncated to {MAX_CHARS} characters to prevent memory issues.")
# >>> End memory optimization <<<

# ----------------------- SUMMARY SETTINGS ----------------------
st.markdown("### ⚙️ Summary Settings")
summary_length = st.select_slider(
    "🧩 Choose Summary Length",
    options=["Short", "Medium", "Detailed"],
    value="Medium"
)

length_map = {
    "Short": (50, 120),
    "Medium": (100, 200),
    "Detailed": (180, 300)
}

# ----------------------- SUMMARIZE -------------------
if st.button("✨ Generate Summary"):
    if not text_input.strip():
        st.warning("⚠️ Please enter or upload some text to summarize.")
    else:
        start_time = time.time()
        st.info("⏳ Initializing model... please wait.")
        progress = st.progress(0)
        for i in range(50):
            time.sleep(0.02)
            progress.progress(i + 1)

        st.write("🔍 Processing your text...")

        # >>> Added for memory optimization <<<
        # Use cached summarizer model and summarize in chunks for long text
        def summarize_in_chunks(text, chunk_size=800):
            summaries = []
            for i in range(0, len(text), chunk_size):
                chunk = text[i:i + chunk_size]
                result = summarizer_model(chunk, max_length=length_map[summary_length][1],
                                          min_length=length_map[summary_length][0],
                                          do_sample=False)[0]['summary_text']
                summaries.append(result)
            return " ".join(summaries)

        summary = summarize_in_chunks(text_input)
        # >>> End memory optimization <<<

        progress.progress(100)
        st.success("✅ Summary generated successfully!")

        end_time = time.time()

        # ------------------- STATS ---------------------
        original_len = len(text_input.split())
        summary_len = len(summary.split())
        reduction = round(100 - (summary_len / original_len * 100), 2) if original_len else 0
        readability = textstat.flesch_reading_ease(summary)

        # Save summary in history
        st.session_state.summaries.append({
            "summary": summary,
            "length": summary_length,
            "original_len": original_len,
            "summary_len": summary_len,
            "reduction": reduction,
            "time": round(end_time - start_time, 2),
            "readability": readability
        })

# ----------------------- HISTORY DISPLAY -------------------
if st.session_state.summaries:
    st.markdown("---")
    st.subheader("🕒 Previous Summaries")

    latest = st.session_state.summaries[-1]
    st.markdown(f"**🧩 Length Mode:** {latest['length']}")
    st.markdown(f"**📄 Words:** {latest['summary_len']} | **📉 Reduced:** {latest['reduction']}%")
    st.markdown(f"**⏱ Time:** {latest['time']}s | **📚 Readability:** {latest['readability']:.2f}")

    with st.expander("📄 Show Latest Summary"):
        st.write(latest["summary"])

    # Compare option
    if len(st.session_state.summaries) > 1:
        st.markdown("### 🔍 Compare with Previous Summary")
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### 🕒 Current Summary")
            st.write(st.session_state.summaries[-1]["summary"])

        with col2:
            st.markdown("#### 🕔 Previous Summary")
            st.write(st.session_state.summaries[-2]["summary"])

    # Clear history
    if st.button("🗑️ Clear Summary History"):
        st.session_state.summaries.clear()
        st.success("🧹 Summary history cleared!")

st.markdown("---")
st.markdown("<p style='text-align:center;'>💡 Built with Streamlit | Hugging Face | PyTorch</p>", unsafe_allow_html=True)
