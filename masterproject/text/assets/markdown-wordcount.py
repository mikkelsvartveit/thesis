import re
import sys

# Read the file
with open(sys.argv[1]) as f:
    content = f.read()

    # Remove LaTeX code blocks within ```{=latex} fence
    content = re.sub(
        r"```{=latex}\n(.*?)\n```",
        "",
        content,
        flags=re.DOTALL,
    )

    # https://github.com/gandreadis/markdown-word-count/blob/master/mwc/counter.py
    # Comments
    content = re.sub(r"<!--(.*?)-->", "", content, flags=re.MULTILINE)
    # Tabs to spaces
    content = content.replace("\t", "    ")
    # More than 1 space to 4 spaces
    content = re.sub(r"[ ]{2,}", "    ", content)
    # Footnotes
    content = re.sub(r"^\[[^]]*\][^(].*", "", content, flags=re.MULTILINE)
    # Indented blocks of code
    content = re.sub(r"^( {4,}[^-*]).*", "", content, flags=re.MULTILINE)
    # Custom header IDs
    content = re.sub(r"{#.*}", "", content)
    # Replace newlines with spaces for uniform handling
    content = content.replace("\n", " ")
    # Remove images
    content = re.sub(r"!\[[^\]]*\]\([^)]*\)", "", content)
    # Remove HTML tags
    content = re.sub(r"</?[^>]*>", "", content)
    # Remove special characters
    content = re.sub(r"[#*`~\-–^=<>+|/:]", "", content)
    # Remove footnote references
    content = re.sub(r"\[[0-9]*\]", "", content)
    # Remove enumerations
    content = re.sub(r"[0-9#]*\.", "", content)

    print(len(content.split()))
