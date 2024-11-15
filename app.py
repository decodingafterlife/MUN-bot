import streamlit as st
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
import torch

# Define your FAQ context here in the backend
FAQ_CONTEXT = """
This is PICT MUN bot.


"""

@st.cache_resource
def load_model():
    """Load the model and tokenizer once and cache them"""
    model = AutoModelForQuestionAnswering.from_pretrained('distilbert-base-cased-distilled-squad')
    tokenizer = AutoTokenizer.from_pretrained('distilbert-base-cased-distilled-squad')
    return model, tokenizer

def answer_question(question, context, model, tokenizer):
    """Get answer for the question from the given context"""
    # Tokenize and encode input text
    inputs = tokenizer.encode_plus(question, context, 
                                 return_tensors="pt", 
                                 max_length=512, 
                                 truncation=True)
    
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
    page_title="PICT MUN Bot",
    page_icon="🤖",
    layout="centered"
)

# Add title and description
st.title("📚 PICT MUN Proccedings Bot")
st.markdown("""
Ask any question about the proceedings:
""")

# Load model and tokenizer
try:
    model, tokenizer = load_model()
except Exception as e:
    st.error(f"Error loading model: {str(e)}")
    st.stop()

# Question input
question = st.text_input(
    "Question",
    placeholder="Ask your question here...",
    help="Enter your question about the proceedings."
)

# Add a button to get the answer
if st.button("Get Answer", type="primary"):
    if not question:
        st.error("Please enter your question!")
    else:
        try:
            with st.spinner("Finding the answer..."):
                answer = answer_question(question, FAQ_CONTEXT, model, tokenizer)
            
            # Display the answer in a nice format
            st.markdown("### Answer")
            st.markdown(f"> {answer}")
            
        except Exception as e:
            st.error("Sorry, I couldn't find an answer to your question. Please try rephrasing or ask something else.")

# Add footer
st.markdown("---")
st.markdown("Made with ❤️ by PICT MUN")
