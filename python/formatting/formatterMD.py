"""
Generate Markdown output, based on the input of split documents trained by 
the neural network
"""
import os
import json
def mdify(input_pairs):
    output = ""
    for i in input_pairs:
        if i["classification"] == 'SPACE':
            output += "\n"
        if i["classification"] == 'LINE':
            output += "---\n"
        if i["classification"] == 'PARAGRAPH':
            string = i['content']
            output += f"{string}\n"
        if i["classification"] == 'HEADER':
            string = i['content'].strip()
            output += f"# {string}\n"
    return output

for i in os.listdir("../../data/3_formatting/3_output/"):
    with open(f"../../data/3_formatting/3_output/{i}") as input_f:
        with open(f'../../data/3_formatting/4_md_output/{i.split(".")[0]}_text.md', 'w') as f:
            print(i)
            pairs = json.loads(input_f.read())
            md_string = mdify(pairs)
            f.write(md_string)
