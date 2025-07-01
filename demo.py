import streamlit as st
from transformers import AutoConfig, AutoModelForSeq2SeqLM, AutoTokenizer
from time import time
import torch
from sample_data import MEDICAL_SAMPLE_DATA

@st.cache_resource
def load_models():
    st_time = time()
    tokenizer = AutoTokenizer.from_pretrained("Babelscape/rebel-large")
    print("+++++ loading Model", time() - st_time)
    model = AutoModelForSeq2SeqLM.from_pretrained("Babelscape/rebel-large")
    if torch.cuda.is_available():
        _ = model.to("cuda:0") # comment if no GPU available
    _ = model.eval()
    print("+++++ loaded model", time() - st_time)
    
    return (tokenizer, model, MEDICAL_SAMPLE_DATA)

def extract_triplets(text):
    triplets = []
    relation = ''
    subject = ''
    object_ = ''
    current = None
    for token in text.split():
        if token == "<triplet>":
            current = 't'
            if relation != '' and subject != '' and object_ != '':
                triplets.append((subject, relation, object_))
                relation = ''
            subject = ''
        elif token == "<subj>":
            current = 's'
            if relation != '' and subject != '' and object_ != '':
                triplets.append((subject, relation, object_))
            object_ = ''
        elif token == "<obj>":
            current = 'o'
            relation = ''
        else:
            if current == 't':
                subject += ' ' + token
            elif current == 's':
                object_ += ' ' + token
            elif current == 'o':
                relation += ' ' + token
    if subject != '' and relation != '' and object_ != '':
        triplets.append((subject, relation, object_))
    return triplets


tokenizer, model, sample_data = load_models()

st.title("🔍 REBEL: Body Structure ↔ Clinical Finding Relations")
st.markdown("""
**Focus on relations between body structures and clinical findings** - Your specific medical use case!

🔗 **Target Relation:**
- `instance of` (411,812 training examples) - **X is an instance/type of Y**

Examples:
- **Clinical Finding**: "There is a mass in the left kidney" → **(mass, instance of, lesion)** + **(left kidney, instance of, organ)**
- **Body Structure**: "Inflammation of the appendix" → **(inflammation, instance of, condition)** + **(appendix, instance of, organ)**
""")

# Add expandable section focused on body structure-clinical finding relations
with st.expander("🏥 Body Structure ↔ Clinical Finding Details"):
    st.markdown("""
    **Target Use Case**: Extract relationships between anatomical structures and clinical findings.
    
    **Body Structures**: left kidney, appendix, right lung, left ankle, liver, femur
    **Clinical Findings**: mass, inflammation, nodule, swelling, fatty infiltration, fracture
    
    **Expected Relations using `instance of`**:
    - **Clinical Findings** → Types: "mass instance of lesion", "inflammation instance of condition"
    - **Body Structures** → Categories: "left kidney instance of organ", "femur instance of bone"
    
    Note: This tests how REBEL classifies both the clinical findings and anatomical structures separately.
    """)
    
    try:
        import pandas as pd
        relations_df = pd.read_csv('data/relations_count.tsv', sep='\t', header=None, names=['Relation', 'Count'])
        instance_relations = relations_df[relations_df['Relation'].str.contains('instance', case=False)]
        st.markdown("**Instance-related relations in REBEL:**")
        for _, row in instance_relations.iterrows():
            st.markdown(f"• `{row['Relation']}` ({row['Count']:,} examples)")
    except Exception as e:
        st.info("Could not load relations file for detailed stats.")

agree = st.checkbox('✏️ Use your own text', False)
if agree:
    text = st.text_input('Input text', 'There is a mass in the left kidney.')
    print(text)
else:
    dataset_example = st.slider('📊 Choose example', 0, len(sample_data)-1, 0)
    text = sample_data[dataset_example]['context']
    
st.markdown("### ⚙️ Model Parameters")
col1, col2, col3 = st.columns(3)
with col1:
    length_penalty = st.slider('Length penalty', 0, 10, 0)
with col2:
    num_beams = st.slider('Number of beams', 1, 20, 3)
with col3:
    num_return_sequences = st.slider('Return sequences', 1, num_beams, 2)

gen_kwargs = {
    "max_length": 256,
    "length_penalty": length_penalty,
    "num_beams": num_beams,
    "num_return_sequences": num_return_sequences,
}

model_inputs = tokenizer(text, max_length=256, padding=True, truncation=True, return_tensors = 'pt')
generated_tokens = model.generate(
    model_inputs["input_ids"].to(model.device),
    attention_mask=model_inputs["attention_mask"].to(model.device),
    **gen_kwargs,
)

decoded_preds = tokenizer.batch_decode(generated_tokens, skip_special_tokens=False)

st.markdown("---")
st.markdown("## 📝 Input Text")
st.info(text)

if not agree:
    st.markdown("## 🎯 Expected Output (Ground Truth)")
    st.code(sample_data[dataset_example]['triplets'])
    expected_triplets = extract_triplets(sample_data[dataset_example]['triplets'])
    st.markdown("**Expected Triplets:**")
    for i, (subj, rel, obj) in enumerate(expected_triplets, 1):
        st.markdown(f"**{i}.** 🔗 **{subj.strip()}** ➜ `{rel.strip()}` ➜ **{obj.strip()}**")

st.markdown("## 🤖 Model Predictions")
decoded_preds = [pred.replace('<s>', '').replace('</s>', '').replace('<pad>', '').strip() for pred in decoded_preds]

for idx, sentence in enumerate(decoded_preds, 1):
    st.markdown(f"### Prediction {idx}")
    st.code(sentence, language="text")
    
    predicted_triplets = extract_triplets(sentence)
    if predicted_triplets:
        st.markdown("**Extracted Triplets:**")
        for i, (subj, rel, obj) in enumerate(predicted_triplets, 1):
            st.markdown(f"**{i}.** 🔗 **{subj.strip()}** ➜ `{rel.strip()}` ➜ **{obj.strip()}**")
    else:
        st.warning("No triplets found in this prediction.")
        
st.markdown("---")
st.markdown("### 📊 Summary")
if not agree:
    expected_count = len(extract_triplets(sample_data[dataset_example]['triplets']))
    st.metric("Expected Triplets", expected_count)

total_predicted = sum(len(extract_triplets(pred)) for pred in decoded_preds)
st.metric("Total Predicted Triplets", total_predicted)
