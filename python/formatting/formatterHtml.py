"""
Generate HTML output, based on the input of split documents trained by 
the neural network
"""

import os
import json
from bs4 import BeautifulSoup
def htmlify(input_pairs):
    output = "<body>"
    output += """<style>
.space {
  margin: 20px;
  padding: 20px;
}
</style>"""
    output += """<style>
.line {
  color: white;
  border: 2px solid black;
  margin: 2px;
  padding: 2px;
}
</style>"""
    output += """<style>
p {
 font-family: verdana;
}
</style>"""
    output += """<style>
h1 {
  font-family: verdana;
  text-align: center;
}
</style>"""
    for i in input_pairs:
        if i["classification"] == 'SPACE':
            output += "<div class=\"space\"></div>"
        if i["classification"] == 'LINE':
            output += "<div class=\"line\"></div>"
        if i["classification"] == 'PARAGRAPH':
            output += f"<p>{i['content']}</p>"
        if i["classification"] == 'HEADER':
            output += f"<h1>{i['content']}</h1>"
    output += "</body>"
    return output

for i in os.listdir("../../data/3_formatting/3_output/"):
    with open(f"../../data/3_formatting/3_output/{i}") as input_f:
        with open(f'../../data/3_formatting/4_html_output/{i.split(".")[0]}_text.html', 'w') as f:
            print(i)
            pairs = json.loads(input_f.read())
            html_string = htmlify(pairs)
            html_output = BeautifulSoup(html_string, 'html.parser').prettify()
            f.write(html_output)
