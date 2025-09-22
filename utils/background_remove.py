# from reportlab.pdfgen import canvas
# from reportlab.lib.pagesizes import letter
# from pdfrw import PdfReader, PdfWriter, PageMerge
# import io

# # Step 1: Create a canvas with 2 images at desired locations
# def create_overlay(image1_path, image2_path):
#     packet = io.BytesIO()
#     can = canvas.Canvas(packet, pagesize=letter)

#     # Place 1st image (e.g., Right Eye)
#     can.drawImage(image1_path, x=50, y=420, width=200, height=200)

#     # Place 2nd image (e.g., Left Eye)
#     can.drawImage(image2_path, x=350, y=420, width=200, height=200)

#     can.save()
#     packet.seek(0)
#     return PdfReader(packet)

# # Step 2: Overlay the image PDF on top of the filled form
# def overlay_images_on_pdf(filled_pdf_path, image1_path, image2_path, output_path):
#     # Load both PDFs
#     overlay_pdf = create_overlay(image1_path, image2_path)
#     base_pdf = PdfReader(filled_pdf_path)
#     output = PdfWriter()

#     # Merge images on the first page
#     for i, page in enumerate(base_pdf.pages):
#         merger = PageMerge(page)
#         if i == 0:
#             merger.add(overlay_pdf.pages[0]).render()
#         output.addpage(page)

#     output.write(output_path)
#     print(f"Final PDF with images saved to: {output_path}")

# # === Usage ===
# overlay_images_on_pdf(
#     filled_pdf_path="filled_temp.pdf",
#     image1_path="r.jpg",
#     image2_path="l.jpg",
#     output_path="final_with_images.pdf"
# )


from PIL import Image
import os
import numpy as np

# Step 1: Convert image to transparent PNG
# def remove_black_background(input_path):
#     image = Image.open(input_path).convert("RGBA")
#     datas = image.getdata()
#     newData = []
#     for item in datas:
#         if item[0] < 20 and item[1] < 20 and item[2] < 20:  # Black threshold
#             newData.append((0, 0, 0, 0))  # Transparent
#         else:
#             newData.append(item)
#     image.putdata(newData)

#     # Save to temporary transparent PNG
#     output_path = input_path.rsplit(".", 1)[0] + "_clean.png"
#     image.save(output_path, "PNG")
#     return output_path

def remove_black_background(input_path, threshold=10):
    image = Image.open(input_path).convert("RGBA")
    np_image = np.array(image)

    # Create a mask for dark background pixels (based on brightness)
    r, g, b, a = np_image[:,:,0], np_image[:,:,1], np_image[:,:,2], np_image[:,:,3]
    brightness = 0.299 * r + 0.587 * g + 0.114 * b
    dark_mask = brightness < threshold

    # Set alpha to 0 for background
    np_image[dark_mask] = [0, 0, 0, 0]

    # Convert back to image
    new_image = Image.fromarray(np_image)

    output_path = input_path.rsplit(".", 1)[0] + "_clean.png"
    new_image.save(output_path, "PNG")
    return output_path



from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from pdfrw import PdfReader, PdfWriter, PageMerge
import io

# Step 2: Overlay the 4 transparent images
def overlay_images(base_pdf_path, image1_path, image1_path2, image2_path, image2_path2, output_path):
    # Clean all 4 images
    clean1 = remove_black_background(image1_path)
    clean1_2 = remove_black_background(image1_path2)
    clean2 = remove_black_background(image2_path)
    clean2_2 = remove_black_background(image2_path2)

    # Create in-memory PDF overlay
    packet = io.BytesIO()
    c = canvas.Canvas(packet, pagesize=letter)

    # Draw images with transparency (mask='auto')
    c.drawImage(clean1_2, x=70, y=505, width=160, height=130, mask='auto')  # Right eye gradcam
    c.drawImage(clean2_2, x=370, y=505, width=160, height=130, mask='auto') # Left eye gradcam
    c.drawImage(clean1, x=70, y=370, width=160, height=130, mask='auto')    # Right eye image
    c.drawImage(clean2, x=370, y=370, width=160, height=130, mask='auto')   # Left eye image

    c.save()
    packet.seek(0)

    # Merge with original PDF
    overlay_pdf = PdfReader(packet)
    base_pdf = PdfReader(base_pdf_path)
    merger = PageMerge(base_pdf.pages[0])
    merger.add(overlay_pdf.pages[0]).render()

    PdfWriter().write(output_path, base_pdf)
    print(f"Overlay complete. Output saved to: {output_path}")