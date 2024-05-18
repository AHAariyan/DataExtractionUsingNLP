# import fitz
# import pytesseract
# from PIL import Image
# import os
# import re
# import json
#
#
# def extract_image_from_image_pdf(pdf_path, output_folder):
#     pdf_document = fitz.open(pdf_path)
#     for page_number in range(len(pdf_document)):
#         page = pdf_document.load_page(page_number)
#         pix = page.get_pixmap()
#         output_image_path = f"{output_folder}/page_{page_number}.png"
#         pix.save(output_image_path)
#     return output_folder
#
#
# def ocr_image_to_text(image_folder):
#     text_data = []
#     for image_file in sorted(os.listdir(image_folder)):
#         if image_file.endswith('.png'):
#             image_path = os.path.join(image_folder, image_file)
#             text = pytesseract.image_to_string(Image.open(image_path), lang='ben')
#             text_data.append(text)
#     return text_data
#
#
# def process_text(text):
#     # Clean up text without removing Bangla characters
#     text = re.sub(r'\s+', ' ', text)
#     text = re.sub(r'\n+', ' ', text)
#     # Remove unwanted ASCII characters but keep Bangla characters
#     text = re.sub(r'[^A-Za-z0-9\u0980-\u09FF\s.,;:!?()-]', '', text)
#     return text.strip()
#
#
# def preprocess_dataset(texts):
#     return [process_text(text) for text in texts]
#
#
# def save_text_data(text, output_file):
#     with open(output_file, 'w') as f:
#         json.dump(text, f, indent=4, ensure_ascii=False)
#
#
# # Paths
# pdf_path = '/media/aariyan/P_seudo Code/Data Collection/Raw-Data/NCTB-Books/Class-9/bangla.pdf'
# image_out_put_folder = '/media/aariyan/P_seudo Code/Data Collection/Extraction/ExtractedImages'
# dataset_path = '/media/aariyan/P_seudo Code/Data Collection/Extraction/DataSet/text_data.json'
#
# # Processing
# print("Extracting images from PDF...")
# extract_image_from_image_pdf(pdf_path=pdf_path, output_folder=image_out_put_folder)
#
# print("Performing OCR on extracted images...")
# text_data = ocr_image_to_text(image_folder=image_out_put_folder)
# print(f"OCR completed. Number of text entries: {len(text_data)}")
#
# print("Preprocessing text data...")
# preprocess_text_data = preprocess_dataset(texts=text_data)
# print(f"Preprocessing completed. Number of preprocessed text entries: {len(preprocess_text_data)}")
#
# print("Saving preprocessed text data...")
# save_text_data(preprocess_text_data, dataset_path)
# print(f"Data saved to {dataset_path}")

import fitz
import pytesseract
from PIL import Image, ImageEnhance, ImageFilter
import os
import re
import json
import cv2
from transformers import pipeline

print("Script started")

# Check paths
pdf_path = '/media/aariyan/P_seudo Code/Data Collection/Raw-Data/NCTB-Books/Class-9/bangla.pdf'
image_out_put_folder = '/media/aariyan/P_seudo Code/Data Collection/Extraction/ExtractedImages'
dataset_path = '/media/aariyan/P_seudo Code/Data Collection/Extraction/DataSet/text_data.json'

print("PDF Path:", pdf_path)
print("Image Output Folder:", image_out_put_folder)
print("Dataset Path:", dataset_path)

# Ensure output directories exist
os.makedirs(image_out_put_folder, exist_ok=True)
os.makedirs(os.path.dirname(dataset_path), exist_ok=True)

print("Directories checked/created successfully")


# def extract_image_from_image_pdf(pdf_path, output_folder, zoom_x=2.0, zoom_y=2.0):
#     print("Entering extract_image_from_image_pdf")
#     try:
#         pdf_document = fitz.open(pdf_path)
#         print(f"PDF opened: {pdf_path}")
#         for page_number in range(len(pdf_document)):
#             page = pdf_document.load_page(page_number)
#             # Zoom to increase resolution
#             mat = fitz.Matrix(zoom_x, zoom_y)
#             pix = page.get_pixmap(matrix=mat)
#             output_image_path = f"{output_folder}/page_{page_number}.png"
#             pix.save(output_image_path)
#             print(f"Extracted image from page {page_number} to {output_image_path}")
#     except Exception as e:
#         print(f"Error extracting images from PDF: {e}")
#     return output_folder


def preprocess_image(image_path):
    print(f"Entering preprocess_image for {image_path}")
    try:
        image = Image.open(image_path)
        # Convert to grayscale
        image = image.convert('L')
        # Enhance contrast
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(2)
        # Apply sharpening filter
        image = image.filter(ImageFilter.SHARPEN)
        preprocessed_image_path = image_path.replace('.png', '_preprocessed.png')
        image.save(preprocessed_image_path)
        print(f"Preprocessed image {preprocessed_image_path}")
        return preprocessed_image_path
    except Exception as e:
        print(f"Error preprocessing image {image_path}: {e}")
        return None


def ocr_image_to_text(image_folder):
    print("Entering ocr_image_to_text")
    text_data = []
    custom_config = r'--oem 3 --psm 6 -l ben'
    for image_file in sorted(os.listdir(image_folder)):
        if image_file.endswith('.png'):
            image_path = os.path.join(image_folder, image_file)
            try:
                preprocessed_image_path = preprocess_image(image_path)
                if preprocessed_image_path:
                    text = pytesseract.image_to_string(Image.open(preprocessed_image_path), config=custom_config)
                    print(f"OCR output for {image_file}: {text}")
                    text_data.append(text)
                else:
                    text_data.append(None)
            except Exception as e:
                print(f"Error performing OCR on image {image_file}: {e}")
                text_data.append(None)
    return text_data


try:
    nlp = pipeline("fill-mask", model="sagorsarker/bangla-bert-base")
    print("Language model loaded successfully")
except Exception as e:
    print(f"Error loading language model: {e}")


def correct_spelling_with_lm(text):
    print("Entering correct_spelling_with_lm")
    try:
        if text is None:
            return None
        tokens = text.split()
        corrected_text = []
        for token in tokens:
            if len(token) > 1:
                corrected_text.append(nlp(f"{token}")[0]['token_str'])
            else:
                corrected_text.append(token)
        return ' '.join(corrected_text)
    except Exception as e:
        print(f"Error correcting spelling: {e}")
        return text


def preprocess_dataset(texts):
    print("Entering preprocess_dataset")
    return [correct_spelling_with_lm(text) for text in texts]


def save_text_data(text, output_file):
    print(f"Entering save_text_data for {output_file}")
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(text, f, indent=4, ensure_ascii=False)
        print(f"Data saved to {output_file}")
    except Exception as e:
        print(f"Error saving text data: {e}")


# Processing
# print("Extracting images from PDF...")
# extract_image_from_image_pdf(pdf_path=pdf_path, output_folder=image_out_put_folder, zoom_x=3.0, zoom_y=3.0)

print("Performing OCR on extracted images...")
text_data = ocr_image_to_text(image_folder=image_out_put_folder)
print(f"OCR completed. Number of text entries: {len(text_data)}")

print("Preprocessing text data...")
preprocess_text_data = preprocess_dataset(texts=text_data)
print(f"Preprocessing completed. Number of preprocessed text entries: {len(preprocess_text_data)}")

print("Saving preprocessed text data...")
save_text_data(preprocess_text_data, dataset_path)
print(f"Data saved to {dataset_path}")
