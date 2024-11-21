# Finetuning veri hazırlığı: Pretraining veri seti ile karşılaştırma ve temel işlemler.

import jsonlines
import itertools
import pandas as pd
from pprint import pprint
​
import datasets
from datasets import load_dataset

# Pretraining veri seti: "The Pile" yerine "Common Crawl" veri seti kullanıldı.
pretrained_dataset = load_dataset("c4", "en", split="train", streaming=True)

# İlk 5 örneği yazdırma.
n = 5
print("Pretrained dataset:")
top_n = itertools.islice(pretrained_dataset, n)
for i in top_n:
  print(i)

# Şirketin finetuning veri seti.
filename = "lamini_docs.jsonl"
instruction_dataset_df = pd.read_json(filename, lines=True)
instruction_dataset_df

# Finetuning veri setinin farklı formatlarda işlenmesi.
examples = instruction_dataset_df.to_dict()
text = examples["question"][0] + examples["answer"][0]
text
if "question" in examples and "answer" in examples:
  text = examples["question"][0] + examples["answer"][0]
elif "instruction" in examples and "response" in examples:
  text = examples["instruction"][0] + examples["response"][0]
elif "input" in examples and "output" in examples:
  text = examples["input"][0] + examples["output"][0]
else:
  text = examples["text"][0]

# Prompt şablonlarının hazırlanması.
prompt_template_qa = """### Question:
{question}
​
### Answer:
{answer}"""
question = examples["question"][0]
answer = examples["answer"][0]
​
text_with_prompt_template = prompt_template_qa.format(question=question, answer=answer)
text_with_prompt_template

prompt_template_q = """### Question:
{question}
​
### Answer:"""

# Finetuning veri setinin prompt formatlarına dönüştürülmesi.
num_examples = len(examples["question"])
finetuning_dataset_text_only = []
finetuning_dataset_question_answer = []
for i in range(num_examples):
  question = examples["question"][i]
  answer = examples["answer"][i]
​
  text_with_prompt_template_qa = prompt_template_qa.format(question=question, answer=answer)
  finetuning_dataset_text_only.append({"text": text_with_prompt_template_qa})
​
  text_with_prompt_template_q = prompt_template_q.format(question=question)
  finetuning_dataset_question_answer.append({"question": text_with_prompt_template_q, "answer": answer})

# Örnek çıktılar.
pprint(finetuning_dataset_text_only[0])
pprint(finetuning_dataset_question_answer[0])

# Finetuning veri setinin saklanması.
with jsonlines.open(f'lamini_docs_processed.jsonl', 'w') as writer:
    writer.write_all(finetuning_dataset_question_answer)

# Finetuning veri setinin yüklenmesi.
finetuning_dataset_name = "lamini/lamini_docs"
finetuning_dataset = load_dataset(finetuning_dataset_name)
print(finetuning_dataset)
