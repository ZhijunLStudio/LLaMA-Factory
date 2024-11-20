import os
import base64
from PIL import Image, ImageDraw, ImageFont
from openai import OpenAI
from mimetypes import guess_type
from tqdm import tqdm
import textwrap  # 用于多行文本处理
import time  # 导入time模块用于计算时间

# 初始化 API 客户端
def initialize_client(api_key, base_url):
    return OpenAI(api_key=api_key, base_url=base_url)

# 将图像编码为 Base64 格式
def encode_image_with_mime(image):
    from io import BytesIO
    img_byte_arr = BytesIO()
    image.save(img_byte_arr, format="PNG")
    base64_data = base64.b64encode(img_byte_arr.getvalue()).decode('utf-8')
    mime_type = "image/png"  # 假设是 PNG 格式
    return f"data:{mime_type};base64,{base64_data}"

def resize_image(image, min_width=28, min_height=28):
    width, height = image.size
    
    # 强制设置最小高度和宽度
    if width < min_width or height < min_height:
        # 计算需要的缩放比例
        scale_factor = max(min_width / width, min_height / height)
        new_width = int(width * scale_factor)
        new_height = int(height * scale_factor)
        # 使用新的宽度和高度进行调整
        return image.resize((new_width, new_height), Image.Resampling.LANCZOS)

    # 如果宽度和高度已经满足要求，则返回原图
    return image

# 计算多行文本的总高度
def get_multiline_text_height(text, font, max_width):
    lines = textwrap.wrap(text, width=max_width)
    total_height = 0
    for line in lines:
        bbox = font.getbbox(line)  # 获取文本的边界框
        total_height += bbox[3] - bbox[1]  # bbox[3]是底部y坐标，bbox[1]是顶部y坐标
    return total_height

# 在图像上方添加文字区域
def add_text_to_image(image, text, font_path="arial.ttf", font_size=20, max_width=60):
    width, height = image.size
    padding = 10

    # 加载字体
    try:
        font = ImageFont.truetype(font_path, font_size)
    except:
        font = ImageFont.load_default()

    # 计算多行文本的总高度
    text_height = get_multiline_text_height(text, font, max_width) + 2 * padding
    new_image = Image.new("RGB", (width, height + text_height), "white")
    new_image.paste(image, (0, text_height))

    draw = ImageDraw.Draw(new_image)
    y_offset = padding
    lines = textwrap.wrap(text, width=max_width)
    for line in lines:
        draw.text((padding, y_offset), line, fill="black", font=font)
        bbox = font.getbbox(line)
        y_offset += bbox[3] - bbox[1]  # 更新y_offset为下一行的起始位置

    return new_image

# 调用 API 进行 OCR 识别并处理错误的逻辑
def perform_ocr(client, image_base64):
    messages = [
        {"role": "system", "content": "请对以下图片进行OCR识别，仅提取文字。"},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "这是一张图片，请提取其中的文字内容。"},
                {"type": "image_url", "image_url": {"url": image_base64}}
            ]
        }
    ]
    
    result = client.chat.completions.create(
        messages=messages,
        model="Qwen2-VL-7B-Instruct",
        max_tokens=300
    )
    
    text = result.choices[0].message.content.strip()
    return text

# 主处理逻辑
def process_images(input_folder, output_folder, client):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for file_name in tqdm(os.listdir(input_folder)):
        file_path = os.path.join(input_folder, file_name)
        if not os.path.isfile(file_path):
            continue

        try:
            start_time = time.time()  # 记录处理开始时间

            # 打开图像
            image = Image.open(file_path)
            image = resize_image(image)  # 调整图像大小

            # 将图像编码为 Base64
            image_base64 = encode_image_with_mime(image)
            
            # 进行 OCR 识别，最多尝试 3 次
            ocr_attempts = 0
            text = ""
            while ocr_attempts < 3:
                text = perform_ocr(client, image_base64)
                
                if text != "这是一张图片，请提取其中的文字内容。":
                    break
                
                ocr_attempts += 1
                print(f"识别结果无效，正在尝试第 {ocr_attempts} 次重试...")

            if text == "这是一张图片，请提取其中的文字内容。":
                text = "problem img"  # 如果三次都无法获取有效文本，返回问题图片

            print(f"识别结果: {text}")

            # 在图像上方添加文字
            processed_image = add_text_to_image(image, text)

            # 保存结果图像
            output_path = os.path.join(output_folder, file_name)
            processed_image.save(output_path)

            end_time = time.time()
            print(f"处理完成: {file_name}, 输出路径: {output_path}")
            print(f"处理单个图像总耗时: {end_time - start_time:.2f}秒")

        except Exception as e:
            print(f"处理文件 {file_name} 时出错: {str(e)}")

if __name__ == '__main__':
    # 配置
    input_folder = "latex_ocr_200"
    output_folder = "latex_ocr_200_output"
    api_key = "0"
    base_url = "http://10.10.7.3:37000/v1"

    # 初始化客户端
    client = initialize_client(api_key, base_url)

    # 处理图像
    process_images(input_folder, output_folder, client)
