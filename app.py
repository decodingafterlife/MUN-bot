import streamlit as st
from transformers import DistilBertTokenizer, DistilBertForQuestionAnswering
import torch

@st.cache_resource
def load_model():
    """Load the model and tokenizer once and cache them"""
    model = DistilBertForQuestionAnswering.from_pretrained('distilbert-base-cased-distilled-squad')
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-cased-distilled-squad')
    return model, tokenizer

def answer_question(question, context, model, tokenizer):
    """Get answer for the question from the given context"""
    # Tokenize and encode input text
    inputs = tokenizer.encode_plus(question, context, return_tensors="pt", max_length=512, truncation=True)
    
    # Make predictions
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Extract start and end scores from the model output
    start_scores = outputs.start_logits
    end_scores = outputs.end_logits
    
    # Find the tokens with the highest start and end scores
    start_index = torch.argmax(start_scores)
    end_index = torch.argmax(end_scores)
    
    # Get the answer span
    answer_tokens = inputs["input_ids"][0][start_index:end_index + 1]
    answer = tokenizer.decode(answer_tokens, skip_special_tokens=True)
    return answer

# Set page configuration
st.set_page_config(
    page_title="FAQ Bot",
    page_icon="🤖",
    layout="wide"
)

# Add title and description
st.title("📚 FAQ Bot")
st.markdown("""
This bot can answer questions based on the context you provide. 
Simply enter your context and question below!
""")

# Load model and tokenizer
model, tokenizer = load_model()

# Create two columns for context and question
col1, col2 = st.columns([2, 1])

with col1:
    # Context input
    context = st.text_area(
        "Context",
        height=300,
        placeholder="Enter the context text here...",
        help="This is the text that the bot will use to find answers to questions."
    )

with col2:
    # Question input
    question = st.text_input(
        "Question",
        placeholder="Ask your question here...",
        help="Enter your question about the context provided."
    )

# Add a button to get the answer
if st.button("Get Answer", type="primary"):
    if not context or not question:
        st.error("Please provide both context and question!")
    else:
        try:
            with st.spinner("Finding the answer..."):
                answer = answer_question(question, context, model, tokenizer)
            
            # Display the answer in a nice format
            st.success("Answer found!")
            st.markdown("### Answer")
            st.markdown(f"> {answer}")
            
            # Add confidence disclaimer
            st.info("""
            Note: The answer is extracted directly from the provided context. 
            Please verify the answer's accuracy for critical applications.
            """)
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")

# Add footer
st.markdown("---")
st.markdown("Made with ❤️ using Streamlit and DistilBERT")