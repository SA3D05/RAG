import json

with open("test_result.json", "r") as f:
    results = json.load(f)


with open("result.md", "a") as out_file:
    for chunk in results:
        with open(chunk["file_path"]) as in_file:
            data = in_file.read()
            # Standard Python slicing: data[start:end] already extracts up to end
            out_file.write(
                data[
                    chunk["first_character_index"] : chunk[
                        "last_character_index"
                    ]
                ]
            )
