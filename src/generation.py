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
            "You are an exact technical Q&A assistant.\n\n"
            "Rules:\n"
            "1. Answer the question directly with the exact \
                command, term, value, or code.\n"
            "2. NEVER say phrases like 'provided in the context', \
                'the text states', 'according to the documents', or \
                    'the context mentions'. State the facts directly.\n"
            "3. Do not describe what the context contains—extract \
                and output the answer itself.\n"
            "4. If the context does not contain the answer, \
                reply ONLY with: "
            "'I cannot answer this question based on the \
                provided documents.'\n"
            "5. Keep the answer to 1-2 direct sentences maximum."
        )

    def __get_chunk_str(self, source: MinimalSource):
        content = Utils.readfile(Path(source.file_path))
        first, last = source.first_character_index, source.last_character_index
        return content[first : last + 1]

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
