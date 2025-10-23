from pdf2image import convert_from_path
import pytesseract

pages = convert_from_path("SPECIAL-INVESTIGATION-BOARD-FINAL-REPORT.pdf", 300)  # Converts all pages to images
text = ""
for page in pages:
    text += pytesseract.image_to_string(page) + "\n\n"  # OCR each page

with open("output.txt", "w") as f:
    f.write(text)