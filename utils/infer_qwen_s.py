import fitz  # PyMuPDF
from transformers import AutoModelForCausalLM, AutoTokenizer



def qwen_extract(cbc_path):


    def extract_text_from_pdf(pdf_path):
        """Extracts text from a PDF file."""
        text = ""
        try:
            with fitz.open(pdf_path) as doc:
                for page in doc:
                    text += page.get_text()
        except FileNotFoundError:
            print(f"Error: File not found at {pdf_path}")
            return None
        except Exception as e:
            print(f"An error occurred: {e}")
            return None
        return text

    # Replace 'your_document.pdf' with the actual path to your PDF file
    pdf_text = extract_text_from_pdf(cbc_path)

    if pdf_text:
        print("Successfully extracted text from the PDF.")
        # You can now use the 'pdf_text' variable with your LLM model
        # For example:
        # llm_model.process_text(pdf_text)
    else:
        print("Failed to extract text from the PDF.")



    model_name = "models/qwen_s"

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    prompt = pdf_text
    messages = [
        {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant. You are given with unstructured pdf report text of cbc. your task is to properly structure it"},
        {"role": "user", "content": prompt}
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=256
    )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]


    return response
