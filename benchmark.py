import json
from itertools import batched
from pathlib import Path

import pandas as pd
import yaml
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, StopStringCriteria, StoppingCriteriaList

from evaluation_metric import score

# %%
MODEL_NAME = "Qwen/Qwen3-0.6B-Base"


class BenchmarkModel:
    DEFAULT_MODEL_NAME = "Qwen/Qwen3-0.6B-Base"
    DEFAULT_EXAMPLES_PATH = "data/examples_extended/part_1/natural_purchase_order.json"

    EXAMPLE_TEMPLATE = (
    "Message: ```{message}```\n"
    "{format}: ```{json_text}```"
    )


    def __init__(self, model_name: str = None, examples_path: str = None, yaml_format: bool = False):
        self.model_name = model_name or self.DEFAULT_MODEL_NAME
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name, torch_dtype="auto", device_map="auto")
        self.stop = StopStringCriteria(self.tokenizer, ["```"])
        self.examples_path = examples_path or self.DEFAULT_EXAMPLES_PATH
        self.examples = self.get_examples()
        self.yaml_format = yaml_format

    def format_entry_for_llm(self, entry: dict, include_json: bool = True) -> str:
        if include_json:
            formated_text = yaml.dump(entry["json_data"]) if self.yaml_format else json.dumps(entry["json_data"], indent=2)
        else:
            formated_text = ""
        formated_text =  self.EXAMPLE_TEMPLATE.format(
            message=entry["natural_language"],
            json_text=formated_text,
            format="YAML" if self.yaml_format else "JSON"
        )
        if not include_json:
            formated_text = formated_text[:-3]
        return formated_text

    def get_examples(self) -> list:
        with open(self.examples_path, "r") as f:
            examples = json.load(f)
        return examples[:5]

    def prompt(self, entry: dict) -> str:
        example_texts = [self.format_entry_for_llm(example) for example in self.examples]
        context_learning_prompt = "\n\n".join(
            example_texts + [self.format_entry_for_llm(entry, include_json=False)]
        )
        return context_learning_prompt

    def sanitize_prediction(self, prediction: str) -> str:
        clean_text = prediction.split("```")[-2]
        if self.yaml_format:
            parsed_yaml = {}
            try:
                parsed_yaml = yaml.safe_load(clean_text)
            except:
                pass
            return json.dumps(parsed_yaml, indent=2)
        return clean_text

    def predict(self, entry: dict) -> str:
        inputs = self.tokenizer(self.prompt(entry), return_tensors="pt").to(self.model.device)
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=500,
            pad_token_id=self.tokenizer.eos_token_id,
            stopping_criteria=StoppingCriteriaList([self.stop])
        )
        full_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return self.sanitize_prediction(full_text)


    def predict_batch(self, batch: list[dict]) -> list[str]:
        prompts = [self.prompt(entry) for entry in batch]
        inputs = self.tokenizer(prompts, return_tensors="pt", padding=True, padding_side='left', truncation=True).to(self.model.device)
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=500,
            pad_token_id=self.tokenizer.eos_token_id,
            stopping_criteria=StoppingCriteriaList([self.stop])
        )
        decoded = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        return [self.sanitize_prediction(full_text) for full_text in decoded]



def evaluate_batch(model: BenchmarkModel, batch: list[dict]) -> tuple[list[int], list[str]]:
    ids = [entry['id'] for entry in batch]
    prediction = model.predict_batch(batch)
    return ids, prediction


def evaluate_model(model: BenchmarkModel, eval_file: Path, output_path: Path = 'benchmark.csv'):
    ids = []
    prediction = []
    batch_size = 8
    with open(eval_file, "r") as f:
        data = json.load(f)
    for batch in tqdm(batched(data, batch_size), total=len(data) // batch_size):
        batch_ids, batch_prediction = evaluate_batch(model, batch)
        ids.extend(batch_ids)
        prediction.extend(batch_prediction)
    df = pd.DataFrame({"id": ids, "prediction": prediction})
    df.to_csv(output_path, index=False)


def main():
    eval = Path("data/competencia/eval.json")
    if not Path("benchmark.csv").exists():
        model = BenchmarkModel()
        evaluate_model(model, eval)
    print(score(
        pd.read_csv("data/competencia/solution.csv"),
        pd.read_csv("benchmark.csv")
    ))

if __name__ == "__main__":
    main()
