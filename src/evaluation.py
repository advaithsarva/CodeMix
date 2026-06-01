from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from sentence_transformers import SentenceTransformer

# Load NLLB-200 for Telugu-English to Hindi Translation
nllb_model = "facebook/nllb-200-distilled-600M"
tokenizer = AutoTokenizer.from_pretrained(nllb_model)
model = AutoModelForSeq2SeqLM.from_pretrained(nllb_model)

# Load Multilingual Sentence Transformer (for Hindi embeddings)
embed_model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

# Function to translate Telugu-English sentences to Hindi
def translate(sentences, src_lang="tel_Latn", tgt_lang="hin_Deva"):
    translations = []
    for sentence in sentences:
        formatted_text = f'>>{src_lang}<< {sentence}'
        inputs = tokenizer(formatted_text, return_tensors="pt")
        output = model.generate(**inputs, forced_bos_token_id=tokenizer.convert_tokens_to_ids(tgt_lang))
        translated_text = tokenizer.decode(output[0], skip_special_tokens=True)
        translations.append(translated_text)
    return translations

# Example sentences
sentences = [ "మన goals ద్వారా ,we can do wonders"]
hindi_sentences = translate(sentences)  # Translate to Hindi

# Generate embeddings for Hindi sentences
hindi_embeddings = embed_model.encode(hindi_sentences)

print(hindi_sentences)  # Hindi translated sentences
print(hindi_embeddings.shape)  # Hindi embeddings (should match original embedding size)
