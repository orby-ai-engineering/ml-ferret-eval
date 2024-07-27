import json
import os
import re
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

def get_high_contrast_color(image, x, y):
    # 获取图像在(x, y)位置的像素颜色
    try:
        pixel = image.getpixel((x, y))
        # 计算红色和蓝色的亮度
        red_brightness = pixel[0]
        blue_brightness = pixel[2]
    except:
        red_brightness = 0
        blue_brightness = 0

    if red_brightness > blue_brightness:
        return "blue"  # 如果红色分量较高，使用蓝色字体
    else:
        return "red"  # 如果蓝色分量较高，使用红色字体

def draw_multiline_text(draw, text, position, font, max_width, fill):
    lines = []
    words = text.split()
    # print('words:', words)
    while words:
        line = ''
        while words:
            test_line = line + words[0] + ' '
            bbox = font.getbbox(test_line)
            # print(f'test_line: "{test_line}", bbox: {bbox}')
            if bbox[2] <= max_width:
                line = test_line
                words.pop(0)
            else:
                break
        # print('line:', line)
        if line.strip():  # 确保 line 不为空
            lines.append(line.strip())
        else:
            # 如果 line 为空，弹出当前的 word 避免死循环
            print(f'Unable to add word "{words[0]}" to line due to exceeding max_width')
            lines.append(words.pop(0))
    
    y = position[1]
    for line in lines:
        draw.text((position[0], y), line, font=font, fill=fill)
        y += font.getbbox(line)[3] - font.getbbox(line)[1]

def extract_coordinates(operation, image_path):
    # Match the operation string for tap and box coordinates
    tap_match = re.search(r'tap\s*\[\[(\d+),(\d+)\]\]', operation, re.IGNORECASE)
    box_match = re.search(r'\[\[(\d+),(\d+),(\d+),(\d+)\]\]', operation)

    image = Image.open(image_path)
    width, height = image.size
    
    if tap_match:
        x, y = map(int, tap_match.groups())
        x = int(width * (x / 1000))
        y = int(height * (y / 1000))
        # print(x, y)
        return x, y
    elif box_match:
        x1, y1, x2, y2 = map(int, box_match.groups())
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        center_x = int(width * (center_x / 1000))
        center_y = int(height * (center_y / 1000))
        # print(center_x, center_y)
        return center_x, center_y
    else:
        # print('hi', operation)
        # raise ValueError("Operation format not recognized", operation)
        return 0, 0

def save_image_with_annotations(data, save_base_dir, correct, coords):
    img_filename = data["img_filename"]
    image_path = os.path.join("/fs/ess/PAS1576/boyu_gou/Benchmark/ScreenSpot/screenspot_imgs_original_size", img_filename)
    image = Image.open(image_path)
    draw = ImageDraw.Draw(image)

    img_width, img_height = image.size
    font_size = int(img_width * 0.02)
    font = ImageFont.load_default(size = font_size) 

    bbox_x, bbox_y, bbox_width, bbox_height = data['bbox']
    bbox_x = int(bbox_x)
    bbox_y = int(bbox_y)
    bbox_width = int(bbox_width)
    bbox_height = int(bbox_height)

    draw.rectangle([bbox_x, bbox_y, bbox_x + bbox_width, bbox_y + bbox_height], outline="blue", width=2)

    x, y = coords
    draw.ellipse([x - 5, y - 5, x + 5, y + 5], fill="red", outline="red")

    text = data.get("description", data.get("next_action", ""))
    max_text_width = image.width - 40  # Assuming 20 pixels padding on each side
    text_color = get_high_contrast_color(image, 20, 20)

    draw_multiline_text(draw, text, (20, 20), font, max_text_width, fill=text_color)

    save_dir = os.path.join(save_base_dir, "true" if correct else "false")
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, os.path.basename(image_path))
    print(save_path)
    image.save(save_path)

def is_output_in_bbox(data):
    bbox = data['bbox']
    output = data.get('output', "")
    print('output', output)
    image_path = os.path.join("/fs/ess/PAS1576/boyu_gou/Benchmark/ScreenSpot/screenspot_imgs_original_size", data["img_filename"])
    
    if not output:
        print('hi', data)
        # print(data['operation'])
        operation = data.get('operation', "")
        if operation:
            x, y = extract_coordinates(data['operation'], image_path)
            if x == 0 and y == 0:
                return False, None
        else:
            return False, None
    else:
        x,y = map(int, output.strip('()').split(', '))

        scale = data.get('scale', 1)
        # print('scale', scale)
        x = x / scale
        y = y / scale

    # Extract the bbox parameters
    bbox_x, bbox_y, bbox_width, bbox_height = bbox

    is_in_bbox = bbox_x <= x <= bbox_x + bbox_width and bbox_y <= y <= bbox_y + bbox_height
    return is_in_bbox, (x, y)

def calculate_accuracy(file_path, save_base_dir, save):
    categories = {
        'mobile_text': {'total': 0, 'correct': 0},
        'mobile_icon': {'total': 0, 'correct': 0},
        'desktop_text': {'total': 0, 'correct': 0},
        'desktop_icon': {'total': 0, 'correct': 0},
        'web_text': {'total': 0, 'correct': 0},
        'web_icon': {'total': 0, 'correct': 0},
    }
    
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            data = json.loads(line)
            key = f"{data['platform']}_{data['data_type']}"
            if key in categories:
                categories[key]['total'] += 1
                correct, coords = is_output_in_bbox(data)
                if correct:
                    categories[key]['correct'] += 1
                if coords and save:
                    save_image_with_annotations(data, save_base_dir, correct, coords)
    
    accuracies = {}
    for key, values in categories.items():
        total = values['total']
        correct = values['correct']
        accuracies[key] = correct / total if total > 0 else 0
    
    average_accuracy = sum(accuracies.values()) / len(accuracies) if accuracies else 0

    print(categories)
    
    return accuracies, average_accuracy

# Replace 'data.jsonl' with the path to your jsonl file
file_path = '/fs/ess/PAS1576/boyu_gou/demi/data/screenspot/results/llava_72b/ans_full_13m_v1.jsonl'
save_base_dir = ''
# save_base_dir = '/fs/ess/PAS1576/boyu_gou/demi/data/screenspot/results/pred_images/gpt4v_seeclick'
save = False
accuracies, average_accuracy = calculate_accuracy(file_path, save_base_dir, save)

# print(file_path)

# Print the accuracies for each category
for category, accuracy in accuracies.items():
    print(f"{category}: {accuracy:.2%}")

# Print the average accuracy
print(f"Average Accuracy: {average_accuracy:.2%}")
