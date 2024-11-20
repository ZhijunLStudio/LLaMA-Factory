
import os
import base64
from openai import OpenAI
import requests
from transformers.utils.versions import require_version
from mimetypes import guess_type  # 用于推断 MIME 类型

require_version("openai>=1.5.0", "To fix: pip install openai>=1.5.0")

def encode_image_with_mime(image_path):
    # 推断图片的 MIME 类型
    mime_type, _ = guess_type(image_path)
    if not mime_type:
        raise ValueError("无法确定图片的 MIME 类型，请检查文件路径或扩展名是否正确。")
    
    # 读取图片并转换为 Base64
    with open(image_path, "rb") as image_file:
        base64_data = base64.b64encode(image_file.read()).decode('utf-8')

    # 构建 Base64 URL
    base64_url = f"data:{mime_type};base64,{base64_data}"
    return base64_url



if __name__ == '__main__':
    port = 37000

    # 初始化 API 客户端
    client = OpenAI(
        api_key="0",  # 如果需要自定义认证，这里改为你的 API Key
        base_url="http://10.10.7.3:{}/v1".format(os.environ.get("API_PORT", port)),
    )

    # 定义提示语
    custom_prompt = "请对以下图片进行OCR识别，仅提取文字。"

    # 本地图片路径
    image_path = "show_databook_images_book_1_image_282_3.jpg"

    # 将本地图片转为base64编码
    image_base64 = encode_image_with_mime(image_path)

    # 构建消息列表
    messages = [
        {
            "role": "system",
            "content": "请对以下图片进行OCR识别，仅提取文字。"
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "这是一张图片，请提取其中的文字内容。"
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": image_base64
                    }
                }
            ]
        }
    ]

    try:
        # 调用API
        result = client.chat.completions.create(
            messages=messages,
            model="Qwen2-VL-7B-Instruct",
            max_tokens=300
        )
        # 打印结果
        print(result.choices[0].message.content)
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        print("Response details:", getattr(e, 'response', None))
