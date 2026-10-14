from loader import load_document


file_path = r"C:\WORK DRIVE\SIH26\data\documents\test.txt"

text = load_document(file_path)

print("=" * 60)
print("DOCUMENT LOADED")
print("=" * 60)
print(text)
print("=" * 60)