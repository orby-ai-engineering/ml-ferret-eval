import io
import base64
import dotenv
import requests
import shortuuid
import argparse
import os
import json
from tqdm import tqdm
from PIL import Image

dotenv.load_dotenv()
DEBUG = os.getenv("DEBUG", False)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
API_HEADERS = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {OPENAI_API_KEY}",
}
OPENAI_API_URL = "https://api.openai.com/v1/chat/completions"

def action_grounding_with_vision_prompt(
    action_descrption: str,
    screenshot_base64: str,
):
    return [
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{screenshot_base64}",
                    },
                },
                {
                    "type": "text",
                    "text": f"""\
In this UI screenshot, what is the position of the element corresponding to the command \"{action_descrption}\"? \
Please answer this question with a pair of coordinates [x, y], each normalized to a value between 0 and 1.\
Please do not output anything else.\
""",
                },
            ]
        }
    ]

def action_grounding_with_gpt4o(
    screenshot_bytes: io.BytesIO,
    action_description: str,
    max_tokens: int = 2000,
    temperature: float = 0,
    debug: bool = True,
) -> dict[str, str]:
    """
    Action grounding with GPT-4o.
    """
    screenshot_base64 = base64.b64encode(screenshot_bytes.getvalue()).decode("utf-8")
    headers = API_HEADERS

    payload = {
        "model": "gpt-4o",
        "messages": action_grounding_with_vision_prompt(
            action_description,
            screenshot_base64,
        ),
        "max_tokens": max_tokens,
        "temperature": temperature,
    }

    if debug:
        messages = [
            msg["text"] if "text" in msg else "<image>"
            for msg in payload["messages"][0]["content"]
        ]
        for msg in messages:
            print(msg, end="")
        print()

    # send the request to GPT
    response = requests.post(
        OPENAI_API_URL,
        headers=headers,
        json=payload,
    )
    if debug:
        print(response.json())
        print()
    
    response_text: str = response.json()["choices"][0]["message"]["content"]
    return response_text

def eval_gpt4o(args):
    questions = [json.loads(q) for q in open(args.question_file, "r")]
    # os.makedirs(os.path.dirname(args.answers_file), exist_ok=True)
    ans_file = open(args.answers_file, "a")

    for line in tqdm(questions):
        if "description" in line:
            line.pop("description")
        image_base_dir = args.image_folder
        image_path = os.path.join(image_base_dir, line[args.image_key])
        
        print(line)

        description = line["instruction"]

        image = Image.open(image_path)
        width, height = image.size
        buf = io.BytesIO()
        image.save(buf, format="PNG")

        response = action_grounding_with_gpt4o(buf, description)

        # Parse the response to get ratio coordinates
        ratio_coords = eval(response.strip())
        x_ratio, y_ratio = ratio_coords
        x_coord = int(x_ratio * width)
        y_coord = int(y_ratio * height)

        print('response', response)
        print('ratio_coords', ratio_coords)
        print('x_coord', x_coord)
        print('y_coord', y_coord)

        ans_id = shortuuid.uuid()
        line["output"] = [x_coord, y_coord]
        line["answer_id"] = ans_id
        line["model_id"] = "gpt-4o"
        line["scale"] = 1.0
        
        ans_file.write(json.dumps(line) + "\n")
        ans_file.flush()

    ans_file.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--answers-file", type=str, default="./4o-answers.jsonl")
    parser.add_argument("--question-file", type=str, default="remaining_question.jsonl")
    parser.add_argument("--image-folder", type=str, default="../screenspot_imgs")
    parser.add_argument("--image-key", type=str, default="img_filename")
    args = parser.parse_args()

    eval_gpt4o(args)
