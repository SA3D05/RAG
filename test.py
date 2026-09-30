import json

with open("README.md") as f:
    content = f.read()


with open("data/processed/chunks.json") as f:
    data = json.loads(f.read())


for d in data:
    print(
        f"{content[d['first_character_index'] : d['last_character_index'] + 1]!r}"
    )



'*This project has been cr'
'een created as part of th'
'um by satifi*\n# RAG