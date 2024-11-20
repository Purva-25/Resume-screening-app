import streamlit as st
import pickle
import PyPDF2
import re

# Load models and encoders
try:
    knn_model = pickle.load(open('clf.pkl', 'rb'))
    tfidf = pickle.load(open('tfidf.pkl', 'rb'))
    le = pickle.load(open('encoder.pkl', 'rb'))
except FileNotFoundError as e:
    st.error(f"Error loading model files: {e}")
    st.stop()

def cleanResume(txt):
    cleanText = re.sub(r'http\S+\s', ' ', txt)
    cleanText = re.sub(r'RT|cc', ' ', cleanText)
    cleanText = re.sub(r'#\S+\s', ' ', cleanText)
    cleanText = re.sub(r'@\S+', '  ', cleanText)
    cleanText = re.sub(r'[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), ' ', cleanText)
    cleanText = re.sub(r'[^\x00-\x7f]', ' ', cleanText)
    cleanText = re.sub(r'\s+', ' ', cleanText)
    return cleanText

def extract_text_from_pdf(file):
    pdf_reader = PyPDF2.PdfReader(file)
    text = ''.join(page.extract_text() for page in pdf_reader.pages)
    return text


def extract_text_from_txt(file):
    try:
        text = file.read().decode('utf-8')
    except UnicodeDecodeError:
        text = file.read().decode('latin-1')
    return text

def handle_file_upload(uploaded_file):
    file_extension = uploaded_file.name.split('.')[-1].lower()
    if file_extension == 'pdf':
        return extract_text_from_pdf(uploaded_file)
    elif file_extension == 'txt':
        return extract_text_from_txt(uploaded_file)
    else:
        raise ValueError("Unsupported file type. Please upload a PDF, DOCX, or TXT file.")

def pred(input_resume):
    cleaned_text = cleanResume(input_resume)
    vectorized_text = tfidf.transform([cleaned_text]).toarray()
    predicted_category = knn_model.predict(vectorized_text)
    return le.inverse_transform(predicted_category)[0]

def main():
    st.set_page_config(page_title="Resume Category Prediction", page_icon="📄")
    st.title("Resume Category Prediction App")
    st.markdown("Upload a resume in **PDF**, **TXT**, or **DOCX** format to predict its job category.")

    uploaded_file = st.file_uploader("Upload a Resume", type=["pdf", "docx", "txt"])

    if uploaded_file is not None:
        try:
            resume_text = handle_file_upload(uploaded_file)
            st.write("✅ Successfully extracted text from the uploaded resume.")

            if st.checkbox("Show extracted text", False):
                st.text_area("Extracted Resume Text", resume_text, height=300)

            st.subheader("Predicted Job Category")
            category = pred(resume_text)
            st.markdown(f"**The predicted job category is: `{category}`**")
        except Exception as e:
            st.error(f"Error: {e}")

if __name__ == "__main__":
    main()
