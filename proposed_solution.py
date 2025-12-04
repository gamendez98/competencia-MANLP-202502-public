import json
from pathlib import Path
from string import Template

import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, DataCollatorForSeq2Seq, \
    EarlyStoppingCallback
from peft import LoraConfig, get_peft_model, PeftModel
from sklearn.model_selection import train_test_split

from benchmark import evaluate_model
from evaluation_metric import score

SCHEMA_SKELETON = """{
  "buyer": {
    "name": string|null,
    "email": string|null,
    "contact": {
       "phone": string|null,
       "alt_email": string|null,
       "preferred_contact": string
    }|null,
    "addresses": [
      {
        "street": string|null,
        "city": string|null,
        "state": string|null,
        "postal_code": string|null,
        "country": string|null
      }
    ]|null
  },
  "purchases": [
    {
      "product_name": string,
      "quantity": number,
      "currency": string|null,
      "discount_code": string|null
    }
  ],
  "shipping": {
    "method": string|null,
    "preferred_by": string|null
  }|null
}"""

SYSTEM_PROMPT = Template("""Eres un asistente experto en extraer y estructurar datos de ordenes de compra enviadas por email.
Tu tarea es convertir la orden de compra en un JSON válido siguiendo este esquema base:
${SCHEMA_SKELETON}
Reglas de extracción y normalización:
- No inventes información.
- Si un dato no aparece en el texto, usa null.
- No modifiques nombres de productos.
- Respeta exactamente la estructura JSON.
- No agregues explicaciones ni comentarios.
- ESCAPADO CRÍTICO: Si un valor de texto contiene comillas dobles ("), DEBES escaparlas con una barra invertida (\\"). 
   Ejemplo incorrecto: "TV 32"" 
   Ejemplo correcto: "TV 32\\""
""")


class SuggestedDataset:
    def __init__(self, dataset_path: Path, tokenizer):
        self.dataset_path = dataset_path
        self.entries = self.load_entries()
        self.tokenizer = tokenizer
        self.train_entries, self.eval_entries = train_test_split(self.entries, test_size=0.2, random_state=42)
        self.train = self.process_entries(self.train_entries)
        self.validation = self.process_entries(self.eval_entries)

    def load_entries(self):
        entries = []
        for file_path in self.dataset_path.glob("*.json"):
            with open(file_path) as f:
                partial_entries = json.load(f)
                entries.extend(partial_entries)
        return entries

    def format_entry(self, entry: dict) -> str:
        system_prompt = SYSTEM_PROMPT.substitute(SCHEMA_SKELETON=SCHEMA_SKELETON)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": entry["natural_language"]},
        ]
        if 'json_data' in entry:
            messages.append({"role": "assistant", "content": json.dumps(entry["json_data"], indent=2)})
        self.tokenizer.apply_chat_template(messages)
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False
        ).replace('<think>\n\n</think>\n', '')

    def process_entries(self, entries: list[dict]):
        return [
            self.tokenizer(
                self.format_entry(e),
                return_tensors="pt", padding=True
            )
            for e in entries
        ]


class SuggestedModel:
    DEFAULT_MODEL_NAME = "Qwen/Qwen3-0.6B-Base"

    def __init__(self, model_name: str = None, dataset_path: Path = Path("data/competencia/train"), adapter_path = None):
        self.model_name = model_name or self.DEFAULT_MODEL_NAME
        self.adapter_path = adapter_path
        self.model, self.tokenizer = self.load_model()
        self.dataset = SuggestedDataset(dataset_path, self.tokenizer)

    def load_model(self):
        model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype="auto",
            device_map="auto"
        )
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
        if self.adapter_path:
            model = PeftModel.from_pretrained(model, self.adapter_path)
        else:
            peft_config = LoraConfig(
                r=16,
                lora_alpha=32,
                lora_dropout=0.05,
                bias="none",
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
            )
            model = get_peft_model(model, peft_config)
        return model, tokenizer

    def train(self, output_dir: Path, final_model_path: Path, seed: int = 42, batch_size=8):
        per_device_train_batch_size = 8
        num_train_samples = len(self.dataset.train)
        steps_per_epoch = num_train_samples // per_device_train_batch_size
        logging_steps = max(1, int(steps_per_epoch * 0.05))
        eval_strategy = "epoch"
        save_strategy = "epoch"
        training_args = TrainingArguments(
            output_dir=str(output_dir),
            seed=seed,
            data_seed=seed,
            num_train_epochs=6,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            gradient_accumulation_steps=4,
            gradient_checkpointing=True,
            learning_rate=3e-4,
            lr_scheduler_type="cosine",
            weight_decay=0.01,
            warmup_ratio=0.03,
            logging_steps=logging_steps,
            eval_strategy=eval_strategy,
            save_strategy=save_strategy,
            save_total_limit=2,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            load_best_model_at_end=True,
            group_by_length=True,
            report_to="none"
        )
        data_collator = DataCollatorForSeq2Seq(
            tokenizer=self.tokenizer,
            pad_to_multiple_of=8,
            padding=True
        )
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.dataset.train,
            eval_dataset=self.dataset.validation,
            tokenizer=self.tokenizer,
            data_collator=data_collator,
            callbacks=[
                EarlyStoppingCallback(
                    early_stopping_patience=3,  # Si no mejora en 3 evaluaciones seguidas, PARA.
                    early_stopping_threshold=0.001  # La mejora debe ser mayor a esto para contar
                )
            ]
        )

        trainer.train()
        self.model = trainer.model
        self.model.save_pretrained(final_model_path)

    def predict(self, entry: dict) -> str:
        entry = self.dataset.process_entries([entry])[0]
        outputs = self.model.generate(
            input_ids=entry[0]["input_ids"],
            attention_mask=entry[0]["attention_mask"],
            max_new_tokens=1000,
            pad_token_id=self.tokenizer.eos_token_id,
            return_dict_in_generate=True,
            output_scores=True,
            num_beams=5,
            early_stopping=True,
        )
        return self.extract_json(self.tokenizer.decode(outputs.sequences[0], skip_special_tokens=True))

    def predict_batch(self, entries: list[dict]) -> list[str]:
        prompts = self.dataset.process_entries(entries)
        inputs = self.tokenizer(prompts, return_tensors="pt", padding=True, padding_side='left', truncation=True).to(self.model.device)
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=500,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        decoded = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        return [self.extract_json(full_text) for full_text in decoded]

    @staticmethod
    def extract_json(text):
        try:
            if "assistant" in text:
                text = text.split("assistant")[-1]

            start = text.find('{')
            end = text.rfind('}') + 1

            return text[start:end] if start != -1 and end != -1 else "{}"
        except:
            return "{}"



def main():
    sm = SuggestedModel()
    sm.train(Path("data/output"), Path("output/final_model.pt"))

    eval = Path("data/competencia/eval.json")
    if not Path("submission.csv").exists():
        evaluate_model(sm, eval)
    print(score(
        pd.read_csv("data/competencia/solution.csv"),
        pd.read_csv("benchmark.csv")
    ))



if __name__ == "__main__":
    main()
