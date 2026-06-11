import json, subprocess, sys

WF = "aisafety-cifar10-attack"
OLD_WORK = r"C:\Users\admin\Desktop\AISafety\_vera\work"
NEW_WORK = r"C:\文件\ME\AISCI\Dr.Researcher\projects\aisafety-cifar10-attack\workspace"
OLD_IMAGES = r"C:\Users\admin\Desktop\AISafety\images"
NEW_IMAGES = r"C:\文件\ME\AISCI\Dr.Researcher\projects\aisafety-cifar10-attack\workspace\data\images"

# Path sub rules (order matters)
SUBS = [
    # eval paths
    (OLD_WORK + r"\eval\\", NEW_WORK + r"\code\eval\\"),
    (OLD_WORK + r"\eval/",  NEW_WORK + r"\code\eval/"),
    # repos
    (OLD_WORK + r"\repos\\", NEW_WORK + r"\code\repos\\"),
    (OLD_WORK + r"\repos/",  NEW_WORK + r"\code\repos/"),
    # adv dirs
    (OLD_WORK + r"\adv_",    NEW_WORK + r"\results\adv_"),
    # generic work root
    (OLD_WORK + "\\",        NEW_WORK + "\\"),
    (OLD_WORK + "/",         NEW_WORK + "/"),
    (OLD_WORK,               NEW_WORK),
    # images
    (OLD_IMAGES, NEW_IMAGES),
]

r = subprocess.run(["conda","run","-n","sci","vera","workflow","show",WF,"--json"],
                   capture_output=True, text=True, encoding="utf-8", errors="replace")
wf = json.loads(r.stdout)

ATTACH = {"prompt-eval":"eval-loop","prompt-lit":"lit-survey","prompt-code":"attack-code"}

for p in wf["prompts"]:
    pid = p["id"]
    text = p["prompt"]
    new_text = text
    for old, new in SUBS:
        new_text = new_text.replace(old, new)
    if new_text == text:
        print(f"{pid}: no changes")
        continue
    # Write new text to temp file
    tmp = rf"C:\文件\ME\AISCI\Dr.Researcher\projects\aisafety-cifar10-attack\workspace\{pid}_prompt.txt"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(new_text)
    print(f"{pid}: updated, writing to {tmp}")

print("done - prompt texts saved to workspace/. Run vera update manually.")
