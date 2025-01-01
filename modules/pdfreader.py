# from utils.clear_text import *
import google.generativeai as genai
import pdfplumber

# Cấu hình GenAI với API Key trực tiếp
api_key = "AIzaSyCNxeBRoCdQumvIplANnAO0Anl7kOeHhkI"  # API Key của bạn
genai.configure(api_key=api_key)
# Khởi tạo mô hình Gemini
model = genai.GenerativeModel('gemini-1.5-flash')

def extract_pdf_text(pdf_path):
    all_text = ""

    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                all_text += text
    
    response = model.generate_content(f"Đọc nội dung {all_text} và lưu lại thông tin không cần phẩn hồi")
    # print(response.text)
    question = f"tôi muốn lấy thông tin tư {all_text} và trả về thông tin fullname, phone, skill, học vấn format lại thành json"
    response = model.generate_content(question)
    if response.text:
        return response.text