import sys,json

print(sys.argv[1])
with open(sys.argv[1]) as inp, open(sys.argv[2], 'w') as out:
    j = json.load(inp)
    if j["nbformat"] >=4:
            for i,cell in enumerate(j["cells"]):
                if cell["cell_type"] == "code" and "sim" in cell["metadata"].get("tags",[]):
                    out.write("#cell "+str(i)+"\n")
                    for line in cell["source"]:
                            out.write(line)
                    out.write('\n\n')
    else:
            for i,cell in enumerate(j["worksheets"][0]["cells"]):
                    out.write("#cell "+str(i)+"\n")
                    for line in cell["input"]:
                            out.write(line)
                    out.write('\n\n')
