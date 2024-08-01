import argparse
import torch
import os
import json
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig
from tqdm import tqdm
import shortuuid
from PIL import Image

def eval_model(args):
    # Model
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen-VL-Chat", trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(args.model_path, device_map="cuda", trust_remote_code=True, bf16=True).eval()
    model.generation_config = GenerationConfig.from_pretrained("Qwen/Qwen-VL-Chat", trust_remote_code=True)

    questions = [json.loads(q) for q in open(os.path.expanduser(args.question_file), "r")]
    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")

    for line in tqdm(questions):
        # 这里要改
        # image_base_dir = "/fs/ess/PAS1576/boyu_gou/Benchmark/ScreenSpot/screenspot_imgs_original_size"
        # image_base_dir = "/fs/ess/PAS1576/boyu_gou/Benchmark/Visual_Web_Bench_Element_Grounding/raw_images" 
        image_base_dir = os.path.expanduser(args.image_folder)  

        # 这里的键也要改
        # image_path = os.path.join(image_base_dir, line['img_filename'])
        image_path = os.path.join(image_base_dir, line[args.image_key])
        print('image_path', image_path)
        
        # print(line)

        description = line["description"]

        image = Image.open(image_path)
        width, height = image.size

        prompt = f"In this UI screenshot, what is the position of the element corresponding to the command \"{description}\" (with point)?"
        print('prompt', prompt)
        
        query = tokenizer.from_list_format([
            {'image': image_path},
            {'text': prompt},
        ])

        with torch.no_grad():
            response, history = model.chat(tokenizer, query=query, history=None)
        
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
        line["output"] = f"({x_coord}, {y_coord})"
        line["answer_id"] = ans_id
        line["model_id"] = os.path.expanduser(args.model_path)
        line["scale"] = 1.0
        
        ans_file.write(json.dumps(line) + "\n")
        ans_file.flush()

    ans_file.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="/fs/ess/PAS1576/boyu_gou/demi/model/SeeClick")
    parser.add_argument("--question-file", type=str, required=True)
    parser.add_argument("--answers-file", type=str, required=True)
    parser.add_argument("--image-folder", type=str, default="../screenspot_imgs")
    parser.add_argument("--image-key", type=str, default="img_filename")
    args = parser.parse_args()

    eval_model(args)
