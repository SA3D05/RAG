from transformers import AutoTokenizer, pipeline
from .models import MinimalAnswer, MinimalSearchResults, MinimalSource


class Model:
    def __init__(self) -> None:
        self.model_id = "Qwen/Qwen3-0.6B"

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)

        self.pipe = pipeline(
            "text-generation",
            model=self.model_id,
            tokenizer=self.tokenizer,
        )

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
        content = ""
        with open(source.file_path) as f:
            content = f.read()
        first, last = source.first_character_index, source.last_character_index
        return content[first : last + 1]

    def generate(self, search_result: MinimalSearchResults) -> MinimalAnswer:
        context = "\n".join(
            [self.__get_chunk_str(i) for i in search_result.retrieved_sources]
        )
        messages = [
            {"role": "system", "content": self.system_prompt},
            {
                "role": "user",
                "content": f"Context:\n{context}\n\
                    \nQuestion:\n{search_result.question}",
            },
        ]

        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )

        outputs = self.pipe(
            prompt,
            return_full_text=False,
            max_new_tokens=512,
        )
        return MinimalAnswer(
            question_id=search_result.question_id,
            question=search_result.question,
            retrieved_sources=search_result.retrieved_sources,
            answer=outputs[0]["generated_text"],
        )
