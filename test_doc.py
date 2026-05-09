"""
Test exactly what analyze_medical_document returns for a medical PDF
"""
import sys, os
sys.path.append('.')

# Create a minimal fake medical PDF bytes (text-mode)
import io
# create a minimal text that mimics a medical lab report
import PyPDF2
from PyPDF2 import PdfWriter

w = PdfWriter()
pg = w.add_blank_page(width=612, height=792)
w.add_metadata({'/Title': 'Lab Test'})

# Just use a plain text string for the test
class FakePdf:
    """Simulate extracted PDF text from a real blood sugar report"""
    CONTENT = """
تحليل الدم
Patient: Ahmed
فحص السكر في الدم A1c: 7.2 %
Glucose Fasting: 130 mg/dL
Cholesterol: 195 mg/dL
"""
    @classmethod
    def as_bytes(cls):
        """Simulate returning raw pdf bytes - we override PyPDF2 read"""
        return b"PDF_STUB"

from models.voice_model import analyze_medical_document

# Monkey-patch PyPDF2.PdfReader to return our test text
import models.voice_model as vm
import PyPDF2

original_pdf_reader = PyPDF2.PdfReader

class MockReader:
    def __init__(self, *args, **kwargs):
        class MockPage:
            def extract_text(self): return FakePdf.CONTENT
        self.pages = [MockPage()]

PyPDF2.PdfReader = MockReader

result = analyze_medical_document(
    doc_bytes=b"stub",
    file_type="application/pdf",
    session_id="test-debug",
    language="Darija"
)

PyPDF2.PdfReader = original_pdf_reader   # restore

print("=== ADVICE TEXT ===")
print(result['advice_text'])
print("=== HAS AUDIO ===", bool(result['advice_audio_base64']))
