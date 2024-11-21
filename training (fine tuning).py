# Training işlemleri: Modeli eğitmek için temel işlemler gerçekleştirilir.

from llama import BasicModelRunner

# Modeli seç ve temel veri kümesini yükle
model = BasicModelRunner("EleutherAI/pythia-410m") 
model.load_data_from_jsonlines("lamini_docs.jsonl", input_key="question", output_key="answer")

# Modeli eğit
model.train(is_public=True)  # Eğitimi başlatır

# Baz modelin yüklenmesi ve veri kümesinin işlenmesi
import os
import lamini
lamini.api_url = os.getenv("POWERML__PRODUCTION__URL")
lamini.api_key = os.getenv("POWERML__PRODUCTION__KEY")
import datasets
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from llama import BasicModelRunner

# Veri kümesini yükle
dataset_name = "lamini_docs.jsonl"
dataset_path = f"/content/{dataset_name}"
use_hf = True

# Model ve tokenizer ayarları
model_name = "EleutherAI/pythia-70m"
training_config = {
    "model": {
        "pretrained_name": model_name,
        "max_length": 2048
    },
    "datasets": {
        "use_hf": use_hf,
        "path": dataset_path
    },
    "verbose": True
}

# Tokenizer'ı yükle ve veri kümesini böl
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

train_dataset, test_dataset = tokenize_and_split_data(training_config, tokenizer)

# Modelin cihaz seçimi ve taşınması
base_model = AutoModelForCausalLM.from_pretrained(model_name)
device_count = torch.cuda.device_count()
device = torch.device("cuda" if device_count > 0 else "cpu")
base_model.to(device)

# İnferans fonksiyonu tanımlama
def inference(text, model, tokenizer, max_input_tokens=1000, max_output_tokens=100):
    # Metni tokenize et
    input_ids = tokenizer.encode(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=max_input_tokens
    )
    # Cevabı üret
    device = model.device
    generated_tokens_with_prompt = model.generate(
        input_ids=input_ids.to(device),
        max_length=max_output_tokens
    )
    # Üretilen cevabı döndür
    return tokenizer.batch_decode(generated_tokens_with_prompt, skip_special_tokens=True)[0][len(text):]

# Eğitim ayarları
max_steps = 3
training_args = TrainingArguments(
    learning_rate=1.0e-5,
    num_train_epochs=1,
    max_steps=max_steps,
    per_device_train_batch_size=1,
    output_dir=f"lamini_docs_{max_steps}_steps",
    save_steps=120,
    eval_steps=120,
    logging_steps=1,
    gradient_accumulation_steps=4,
    evaluation_strategy="steps",
    load_best_model_at_end=True,
    save_total_limit=1,
    metric_for_best_model="eval_loss",
    greater_is_better=False
)

# Eğitimi başlat
trainer = Trainer(
    model=base_model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset
)
trainer.train()

# Modeli kaydet
save_dir = f'{training_args.output_dir}/final'
trainer.save_model(save_dir)

# Eğitilmiş modeli yükle ve test et
finetuned_model = AutoModelForCausalLM.from_pretrained(save_dir, local_files_only=True)
finetuned_model.to(device)

test_question = test_dataset[0]['question']
print("Finetuned model's answer: ")
print(inference(test_question, finetuned_model, tokenizer))
