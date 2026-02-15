import fitz  # PyMuPDF

pdf_path = "./docs/sample.pdf"  # change if needed
doc = fitz.open(pdf_path)

print("\n" + "="*60)
print("📄 PDF ANALYSIS")
print("="*60 + "\n")

pages_to_check = min(3, len(doc))

total_text_len = 0
total_images = 0
has_corruption = False

for page_num in range(pages_to_check):
    page = doc[page_num]

    text = page.get_text("text") or ""
    images = page.get_images() or []

    total_text_len += len(text)
    total_images += len(images)

    if "�" in text:
        has_corruption = True

    print(f"Page {page_num + 1}:")
    print(f"  Text length: {len(text)} characters")
    print(f"  Number of images: {len(images)}")
    preview = text[:200].replace("\n", " ")
    print(f"  Text preview: {preview}")
    print()

print("\n" + "="*60)
print("DIAGNOSIS:")

avg_text_len = total_text_len / max(1, pages_to_check)

if avg_text_len < 100:
    print("❌ Very little text found - likely a SCANNED/IMAGE-based PDF")
    print("✅ Suggestion: OCR pipeline needed, or use a text-based PDF")
elif has_corruption:
    print("⚠️ Corrupted characters detected (�)")
    print("✅ Suggestion: try different extraction mode or a cleaner PDF")
else:
    print("✅ Looks like a text-based PDF - extraction should work fine")

print("="*60 + "\n")
