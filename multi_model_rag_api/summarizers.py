import os
import asyncio
import base64
from PIL import Image
import io
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

async def summarize_element(element):
    """Summarize a single element using OpenAI API"""
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0,
        messages=[
            {"role": "system", "content": "You are an assistant tasked with summarizing tables and text for retrieval."},
            {"role": "user", "content": f"You are an assistant tasked with summarizing tables and text for retrieval. These summaries will be embedded and used to retrieve the raw text or table elements. Give a concise summary of the table or text that is well optimized for retrieval. Table or text: {element}"}
        ]
    )
    return response.choices[0].message.content

async def process_batch(elements, max_concurrency=5):
    """Process a batch of elements with concurrency control"""
    semaphore = asyncio.Semaphore(max_concurrency)
    
    async def process_with_semaphore(element):
        async with semaphore:
            return await summarize_element(element)
    
    tasks = [process_with_semaphore(element) for element in elements]
    return await asyncio.gather(*tasks)

def generate_text_summaries(texts, tables, summarize_texts=False):
    """Summarize text elements"""
    # Initialize empty summaries
    text_summaries = []
    table_summaries = []

    # Create event loop for async operations
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    # Apply to text if texts are provided and summarization is requested
    if texts and summarize_texts:
        text_summaries = loop.run_until_complete(process_batch(texts, max_concurrency=5))
    elif texts:
        text_summaries = texts

    # Apply to tables if tables are provided
    if tables:
        table_summaries = loop.run_until_complete(process_batch(tables, max_concurrency=5))

    loop.close()
    return text_summaries, table_summaries

def encode_image(image_path):
    """Convert image to base64 string"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

def resize_base64_image(base64_string, size=(1300, 600)):
    """Resize a base64 encoded image"""
    # Decode base64 string to image
    img_data = base64.b64decode(base64_string)
    img = Image.open(io.BytesIO(img_data))
    
    # Resize image
    img = img.resize(size, Image.Resampling.LANCZOS)
    
    # Convert back to base64
    buffered = io.BytesIO()
    img.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

def image_summarize(img_base64, prompt):
    """Generate image summary using OpenAI API"""
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        max_tokens=1024,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{img_base64}"},
                    },
                ]
            }
        ]
    )
    return response.choices[0].message.content

def generate_img_summaries(path):
    """Generate summaries and base64 encoded strings for images"""
    # Store base64 encoded images
    img_base64_list = []
    # Store image summaries
    image_summaries = []

    # Prompt for image summarization
    prompt = """You are an assistant tasked with summarizing images for retrieval. \
    These summaries will be embedded and used to retrieve the raw image. \
    Give a concise summary of the image that is well optimized for retrieval."""

    # Process each image in the directory
    for img_file in sorted(os.listdir(path)):
        if img_file.endswith(".jpg"):
            img_path = os.path.join(path, img_file)
            base64_image = encode_image(img_path)
            # Resize image to optimize for API
            base64_image = resize_base64_image(base64_image)
            img_base64_list.append(base64_image)
            image_summaries.append(image_summarize(base64_image, prompt))

    return img_base64_list, image_summaries 