import os

from pypdf import PdfReader

import pytesseract
from PIL import Image


def extract_text_from_pdf(
    file_path
):

    text = ""

    reader = PdfReader(file_path)

    for page in reader.pages:

        page_text = page.extract_text()

        if page_text:
            text += page_text + "\n"

    return text.strip()


def extract_text_from_image(
    file_path
):

    image = Image.open(file_path)

    text = pytesseract.image_to_string(
        image
    )

    return text.strip()


def extract_text(file_path):

    extension = os.path.splitext(
        file_path
    )[1].lower()

    if extension == ".pdf":

        return extract_text_from_pdf(
            file_path
        )

    elif extension in [
        ".png",
        ".jpg",
        ".jpeg"
    ]:

        return extract_text_from_image(
            file_path
        )

    else:

        raise ValueError(
            "Unsupported document format."
        )