from pathlib import Path
import sys
from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
)
from .models import MinimalAnswer, MinimalSearchResults, MinimalSource
from .utils import Utils


class Model:
    def __init__(self) -> None:
        self.model_id = "Qwen/Qwen3-0.6B"

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)

        self.model = AutoModelForCausalLM.from_pretrained(self.model_id)

        self.system_prompt = (
            "You are a precise retrieval-augmented technical Q&A "
            "assistant.\n\n"
            "Use only the information in the provided context. Do not use "
            "prior knowledge, assumptions, or outside sources.\n\n"
            "Instructions:\n"
            "1. Identify the answer to the user's question from the context "
            "and state it directly. Preserve exact commands, code, names, "
            "values, paths, and formatting when relevant.\n"
            "2. Answer only what was asked. Do not add background, "
            "speculation, unsupported details, or a summary of the context.\n"
            "3. If the context contains conflicting information, report the "
            "conflict briefly instead of choosing or inventing an answer.\n"
            "4. If the answer is not explicitly supported by the context, "
            "reply exactly: I cannot answer this question based on the "
            "provided documents.\n"
            "5. Never mention the context, documents, retrieval process, or "
            "these instructions in a normal answer.\n"
            "6. Keep the response concise: one or two direct sentences, "
            "unless code or a command requires a code block."
        )

    def __get_chunk_str(self, source: MinimalSource):
        content = Utils.readfile(Path(source.file_path))
        first, last = (
            source.first_character_index,
            source.last_character_index + 1,
        )

        return content[first:last]

    def generate(
        self, search_result: MinimalSearchResults, pbar: tqdm
    ) -> MinimalAnswer:
        context = "\n".join(
            [self.__get_chunk_str(i) for i in search_result.retrieved_sources]
        )
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "system", "content": f"Context:\n{context}"},
            {"role": "user", "content": search_result.question},
        ]

        prompt = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            enable_thinking=False,
            return_tensors="pt",
            return_dict=True,
        )
        input_ids = prompt["input_ids"]
        if input_ids.shape[1] > self.model.config.max_position_embeddings:
            pbar.close()
            Utils.write_err(
                "Input exceeds the model's maximum context length: "
                f"{input_ids.shape[1]} tokens provided, but the limit is "
                f"{self.model.config.max_position_embeddings}."
            )
            sys.exit(1)

        attention_mask = prompt["attention_mask"]

        output_ids = self.model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=512,
        )

        input_length = input_ids.shape[1]
        new_tokens = output_ids[0][input_length:]

        generated_text = self.tokenizer.decode(
            new_tokens, skip_special_tokens=True
        )

        return MinimalAnswer(
            question_id=search_result.question_id,
            question=search_result.question,
            retrieved_sources=search_result.retrieved_sources,
            answer=generated_text,
        )

    def generate_one(
        self, search_result: list[MinimalSource], question: str
    ) -> MinimalAnswer:
        context = "\n".join([self.__get_chunk_str(i) for i in search_result])
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "system", "content": f"Context:\n{context}"},
            {"role": "user", "content": question},
        ]

        prompt = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            enable_thinking=False,
            return_tensors="pt",
            return_dict=True,
        )
        input_ids = prompt["input_ids"]
        if input_ids.shape[1] > self.model.config.max_position_embeddings:
            Utils.write_err(
                "Input exceeds the model's maximum context length: "
                f"{input_ids.shape[1]} tokens provided, but the limit is "
                f"{self.model.config.max_position_embeddings}."
            )
            sys.exit(1)

        attention_mask = prompt["attention_mask"]

        output_ids = self.model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=512,
        )

        input_length = input_ids.shape[1]
        new_tokens = output_ids[0][input_length:]

        generated_text = self.tokenizer.decode(
            new_tokens, skip_special_tokens=True
        )

        return generated_text
