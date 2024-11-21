import datasets
import logging
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

# Logger ayarı
logger = logging.getLogger(__name__)
global_config = None

# Lamini datasetini yükle
dataset = datasets.load_dataset("lamini/lamini_docs")
test_dataset = dataset["test"]

# İlk test sorusu ve cevabını yazdır
print(test_dataset[0]["question"])
print(test_dataset[0]["answer"])

# Model ve tokenizer yükleniyor
model_name = "lamini/lamini_docs_finetuned"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Modeli değerlendirme moduna geçirme
model.eval()

# Basit bir metin karşılaştırma fonksiyonu
def is_exact_match(a, b):
    # İki metin arasında birebir eşleşme olup olmadığını kontrol eder
    return a.strip() == b.strip()

# Metni işleme ve modelden cevap alma fonksiyonu
def inference(text, model, tokenizer, max_input_tokens=1000, max_output_tokens=100):
    # Tokenize işlemi
    tokenizer.pad_token = tokenizer.eos_token
    input_ids = tokenizer.encode(
        text,
        return_tensors="pt",  # Girdiyi PyTorch tensörü formatında döndür
        truncation=True,
        max_length=max_input_tokens
    )

    # Cevap üretme
    device = model.device
    generated_tokens_with_prompt = model.generate(
        input_ids=input_ids.to(device),
        max_length=max_output_tokens
    )

    # Tokenleri metne dönüştürme
    generated_text_with_prompt = tokenizer.batch_decode(generated_tokens_with_prompt, skip_special_tokens=True)

    # Soru metnini cevaptan çıkar
    generated_text_answer = generated_text_with_prompt[0][len(text):]

    return generated_text_answer

# İlk test sorusu için model çalıştırma ve beklenen cevapla karşılaştırma
test_question = test_dataset[0]["question"]
generated_answer = inference(test_question, model, tokenizer)
print(test_question)
print(generated_answer)
answer = test_dataset[0]["answer"]
print(answer)
exact_match = is_exact_match(generated_answer, answer)
print(exact_match)

# Tüm dataset üzerinde değerlendirme
n = 10  # İşlenecek veri sayısı (-1 tüm veri)
metrics = {'exact_matches': []}
predictions = []

for i, item in tqdm(enumerate(test_dataset)):
    question = item['question']
    answer = item['answer']

    try:
        # Model tahmini al
        predicted_answer = inference(question, model, tokenizer)
    except:
        continue
    
    predictions.append([predicted_answer, answer])
    # Doğruluk kontrolü
    exact_match = is_exact_match(predicted_answer, answer)
    metrics['exact_matches'].append(exact_match)

    # İşleme sınırı
    if i > n and n != -1:
        break

# Eşleşen cevapların sayısını yazdır
print('Number of exact matches: ', sum(metrics['exact_matches']))

# Tahminler ve hedef cevapları bir dataframe'e kaydet
df = pd.DataFrame(predictions, columns=["predicted_answer", "target_answer"])
print(df)
