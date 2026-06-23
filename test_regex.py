import re, json

text = """✅ 1. 麻婆豆腐（虽归类为"荤菜"，但传统上常配米饭作主菜）
✅ 2. 葱油拌面（明确分类为"主食"，沪菜风味）
✅ 3. 酸辣蕨根粉（分类为"主食,凉菜"，川味开胃）"""

print("=== Test: checkmark detection ===")
for line in text.split('\n'):
    print(f"  line={line!r}")
    print(f"  '✅' in line = {'✅' in line}")
    has_unicode_check = '\u2705' in line
    print(f"  '\\u2705' in line = {has_unicode_check}")
    clean = re.sub(r'^\s*.\s*(?:\d+[.、．]\s*)?', '', line).strip()
    print(f"  clean={clean!r}")
    m = re.match(r'([\u4e00-\u9fff]{2,12})[（(]', clean)
    if m:
        print(f"  → {m.group(1)}")
    else:
        # Try alternative: match any non-ASCII check-like char
        clean2 = re.sub(r'^\s*[^\w\s]+\s*(?:\d+[.、．]\s*)?', '', line).strip()
        print(f"  clean2={clean2!r}")
        m2 = re.match(r'([\u4e00-\u9fff]{2,12})[（(]', clean2)
        if m2:
            print(f"  → {m2.group(1)}")
        else:
            print("  → NO MATCH")

print("\n=== Test: just split and match ===")
for line in text.split('\n'):
    # Simple approach: find any Chinese name followed by （
    m = re.search(r'[\u4e00-\u9fff]{2,12}[（(]', line)
    if m:
        name = re.match(r'([\u4e00-\u9fff]{2,12})', m.group()).group(1)
        print(f"  → {name}")
