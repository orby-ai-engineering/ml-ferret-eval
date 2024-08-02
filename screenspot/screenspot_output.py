"""
Usage:
--data_path: path of refcoco annotation.
--image_path:  path of refcoco images.
--answers-file: path of output result.

Example:
CUDA_VISIBLE_DEVICES=0 python -m ferret.eval.model_refcoco \
    --model-path checkpoints/ferret_13b/checkpoint-final \
    --image_path data/refcoco/train2014 \
    --data_path data/annotations/finetune_refcocog_test.json \
    --answers-file refexp_result/finetune_refcocog_test \
    --add_region_feature \
    --chunk-idx 0 \
    --num-chunks 1

"""

import argparse
import torch
import os
import json
from tqdm import tqdm

from ferret.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria
from ferret.model.builder import load_pretrained_model
from ferret.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from ferret.conversation import conv_templates, SeparatorStyle
from ferret.utils import disable_torch_init
import re
import math
import numpy as np

# Added for visualization
from PIL import Image, ImageDraw, ImageFont

VOCAB_IMAGE_W = 1000
VOCAB_IMAGE_H = 1000
DEFAULT_REGION_FEA_TOKEN = "<region_fea>"


def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


def plot_refexp(img, boxes, entities, mode='pred'):
    if mode == "gt":
        color = "green"
    elif mode == "pred":
        color = "blue"
    draw = ImageDraw.Draw(img)
    fnt = ImageFont.load_default()
    draw.rectangle([boxes[0], boxes[1], boxes[2], boxes[3]], outline=color)
    draw.text((boxes[0], boxes[1]-5), f'{entities}', font=fnt, fill=color)
    return img


def remove_punctuation(text: str) -> str:
    punct = [',',]
    for p in punct:
        text = text.replace(p, '')
    return text.strip()


def resize_bbox(box, image_w=None, image_h=None):
    ratio_w = image_w * 1.0 / VOCAB_IMAGE_W
    ratio_h = image_h * 1.0 / VOCAB_IMAGE_H

    new_box = [int(box[0] * ratio_w), int(box[1] * ratio_h), \
               int(box[2] * ratio_w), int(box[3] * ratio_h)]
    return new_box


def find_bbox_template(text, img_w, img_h):
    pattern = r'\[(\d+), (\d+), (\d+), (\d+)\]'
    matches = re.findall(pattern, text)
    new_bboxes = []
    old_bboxes = []
    for match in matches:
        x1, y1, x2, y2 = map(int, match)
        new_box = resize_bbox([x1, y1, x2, y2], img_w, img_h)
        new_bboxes.append(new_box)
        old_bboxes.append([x1, y1, x2, y2])

    set_old_bboxes = sorted(set(map(tuple, old_bboxes)), key=list(map(tuple, old_bboxes)).index)
    list_old_bboxes = list(map(list, set_old_bboxes))

    set_bboxes = sorted(set(map(tuple, new_bboxes)), key=list(map(tuple, new_bboxes)).index)
    list_bboxes = list(map(list, set_bboxes))

    for i in range(len(list_bboxes)):
        x1, y1, x2, y2 = list_old_bboxes[i]
        text = text.replace('[{}, {}, {}, {}]'.format(x1, y1, x2, y2), '')

    if text.endswith(" ."):
        text = text[:-2]
    split_text = text.split(" . ")
    entities = [item.strip() for item in split_text if item.strip() != '']

    return entities, list_bboxes

class ScreenSpotGrounding():
    def __init__(self, img_folder, ann_file):
        self.img_folder = img_folder
        with open(ann_file, "r") as f:
            self.targets = [json.loads(line) for line in f]
        print("NUMBER OF TARGETS: ", len(self.targets))
        self.question_prompt = """\
In this UI screenshot, what is the bounding box of the element corresponding to \"<obj>\"?\
Please make the bounding box as specific as possible.\
"""

    def __getitem__(self, idx):
        img = Image.open(os.path.join(self.img_folder, self.targets[idx]["img_filename"]))
        target = self.targets[idx]
        print("TARGET: ", target)

        bbox_xywh = target["bbox"]
        bbox_xyxy = np.array([bbox_xywh[0], bbox_xywh[1], bbox_xywh[0] + bbox_xywh[2], bbox_xywh[1] + bbox_xywh[3]])
        w, h = img.size
        bbox_xyxy[0::2].clip(min=0, max=w)
        bbox_xyxy[1::2].clip(min=0, max=h)

        assert "<obj>" in self.question_prompt
        question = self.question_prompt.replace("<obj>", target["instruction"])

        target["img_w"] = w
        target["img_h"] = h
        target["question"] = question
        target["bbox_xyxy"] = bbox_xyxy.tolist()

        return img, target
    
    def __len__(self):
        return len(self.targets)

def create_center_point(bbox):
    x1, y1, x2, y2 = bbox
    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2
    return [center_x, center_y]

def eval_model_screenspot(args):
    # Data
    dataset = ScreenSpotGrounding(img_folder=args.image_path,
                                  ann_file=args.data_path)
    data_ids = range(len(dataset))

    # Model
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, _ = load_pretrained_model(model_path, args.model_base, model_name)

    chunk_data_ids = get_chunk(data_ids, args.num_chunks, args.chunk_idx)
    answers_file = os.path.expanduser(args.answers_file)
    ans_file = open(answers_file, "w")

    for i, id in enumerate(tqdm(chunk_data_ids)):
        img, ann = dataset[id]
        qs = ann["question"]
        cur_prompt = qs

        # Plot GTs
        # img = plot_refexp(img, ann["bbox_xyxy"], [ann["description"]], mode="gt")

        if model.config.mm_use_im_start_end:
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

        conv = conv_templates[args.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        print("PROMPT:", prompt)

        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
        image_tensor = image_processor.preprocess(img, return_tensors='pt', do_resize=True,
                                                  do_center_crop=False, size=[args.image_h, args.image_w])['pixel_values'][0]
        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=image_tensor.unsqueeze(0).half().cuda(),
                do_sample=True,
                temperature=args.temperature,
                top_p=args.top_p,
                num_beams=args.num_beams,
                max_new_tokens=1024,
                use_cache=True,
                stopping_criteria=[stopping_criteria],
            )

        input_token_len = input_ids.shape[1]
        n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
        if n_diff_input_output > 0:
            print(f'[Warning] Sample {i}: {n_diff_input_output} output_ids are not the same as the input_ids')
        outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
        outputs = outputs.strip()
        if outputs.endswith(stop_str):
            outputs = outputs[:-len(stop_str)]
        outputs = outputs.strip()
        print("OUTPUTS:", outputs)

        # Plot Preds
        try:
            pred_entities, pred_bboxes = find_bbox_template(outputs, img_w=ann["img_w"], img_h=ann["img_h"])
            img = plot_refexp(img, pred_bboxes[0], pred_entities, mode="pred")
            img.save('refexp_result/images/{}.png'.format(i))
            # create center point output
            ann["output"] = create_center_point(pred_bboxes[0])
            print("PREDICTIONS:", ann["output"])
        except Exception as _:
            # if there is no output, set the output to -1, -1
            print("PREDICTIONS:", "NO BOUNDING BOX FOUND")
            ann["output"] = [-1, -1]
        ann["prompt"] = cur_prompt
        ann["operation"] = outputs
        ans_file.write(json.dumps(ann) + "\n")
        ans_file.flush()
    ans_file.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image_path", type=str, default="data/refcoco/train2014")
    parser.add_argument("--data_path", type=str, default="data/annotations/finetune_refcoco_testA.json")
    parser.add_argument("--answers-file", type=str, default="refexp_result/refcoco_testA")
    parser.add_argument("--conv-mode", type=str, default="ferret_v1")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--image_w", type=int, default=336)  #  224
    parser.add_argument("--image_h", type=int, default=336)  #  224
    parser.add_argument("--temperature", type=float, default=0.001)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    args = parser.parse_args()

    eval_model_screenspot(args)
