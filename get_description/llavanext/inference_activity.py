import argparse
import time
import torch

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import process_anyres_image, tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria

import json
import os
import math
from tqdm import tqdm
from decord import VideoReader, cpu

from transformers import AutoConfig

import cv2
import base64
import openai

from PIL import Image

import numpy as np

def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i : i + chunk_size] for i in range(0, len(lst), chunk_size)]

def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]

def parse_args():
    """
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser()

    # Existing command-line arguments
    parser.add_argument("--model-path", type=str, default="")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--conv-mode", type=str, default="vicuna_v1")
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--mm_resampler_type", type=str, default="spatial_pool")
    parser.add_argument("--mm_spatial_pool_stride", type=int, default=2)
    parser.add_argument("--mm_spatial_pool_out_channels", type=int, default=1024)
    parser.add_argument("--mm_spatial_pool_mode", type=str, default="average")
    parser.add_argument("--image_aspect_ratio", type=str, default="anyres")
    parser.add_argument("--image_grid_pinpoints", type=str, default="[(224, 448), (224, 672), (224, 896), (448, 448), (448, 224), (672, 224), (896, 224)]")
    parser.add_argument("--mm_patch_merge_type", type=str, default="spatial_unpad")
    parser.add_argument("--overwrite", type=lambda x: (str(x).lower() == 'true'), default=True)
    parser.add_argument("--for_get_frames_num", type=int, default=32)
    parser.add_argument("--load_8bit",  type=lambda x: (str(x).lower() == 'true'), default=False)
    parser.add_argument("--prompt", type=str, default="Please provide a detailed description of the video, focusing on the main subjects, their actions, the background scenes.")
    parser.add_argument("--api_key", type=str, help="OpenAI API key")
    parser.add_argument("--mm_newline_position", type=str, default="grid")
    parser.add_argument("--force_sample", type=lambda x: (str(x).lower() == 'true'), default=False)
    parser.add_argument("--add_time_instruction", type=str, default=False)

    # New command-line arguments
    parser.add_argument("--video_folder", type=str, default='', help="Directory containing video files.")
    parser.add_argument("--input_json", type=str, default='', help="Path to the input JSON file.")
    parser.add_argument("--output_results", type=str, default='', help="Path to the output results JSON file.")

    return parser.parse_args()

def load_video(video_path, args):
    if args.for_get_frames_num == 0:
        return np.zeros((1, 336, 336, 3)), "", 0.0
    vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
    total_frame_num = len(vr)
    video_time = total_frame_num / vr.get_avg_fps()
    fps = round(vr.get_avg_fps())
    frame_idx = [i for i in range(0, len(vr), fps)]
    frame_time = [i / fps for i in frame_idx]
    if len(frame_idx) > args.for_get_frames_num or args.force_sample:
        sample_fps = args.for_get_frames_num
        uniform_sampled_frames = np.linspace(0, total_frame_num - 1, sample_fps, dtype=int)
        frame_idx = uniform_sampled_frames.tolist()
        frame_time = [i / vr.get_avg_fps() for i in frame_idx]
    frame_time = ",".join([f"{i:.2f}s" for i in frame_time])
    spare_frames = vr.get_batch(frame_idx).asnumpy()
    return spare_frames, frame_time, video_time


def load_video_base64(path):
    video = cv2.VideoCapture(path)

    base64Frames = []
    while video.isOpened():
        success, frame = video.read()
        if not success:
            break
        _, buffer = cv2.imencode(".jpg", frame)
        base64Frames.append(base64.b64encode(buffer).decode("utf-8"))

    video.release()
    # print(len(base64Frames), "frames read.")
    return base64Frames


def find_video_file(video_folder, video_name):
    """
    Try to find the video file in the folder by prepending 'v_' or 'v__' to the video_name.
    Returns the full path if found, else None.
    """
    possible_prefixes = ["v_", "v__"]
    possible_extensions = [".mp4", ".avi", ".mov", ".mkv"]  # Add more extensions if needed

    for prefix in possible_prefixes:
        for ext in possible_extensions:
            filename = f"{prefix}{video_name}{ext}"
            full_path = os.path.join(video_folder, filename)
            if os.path.exists(full_path):
                return full_path
    return None


def run_inference(args):
    """
    Run inference based on the input JSON and save results to output_results.
    """
    # Initialize the model
    if "gpt4v" != args.model_path:
        model_name = get_model_name_from_path(args.model_path)
        # Set model configuration parameters if they exist
        if args.overwrite:
            overwrite_config = {}
            overwrite_config["mm_spatial_pool_mode"] = args.mm_spatial_pool_mode
            overwrite_config["mm_spatial_pool_stride"] = args.mm_spatial_pool_stride
            overwrite_config["mm_newline_position"] = args.mm_newline_position

            cfg_pretrained = AutoConfig.from_pretrained(args.model_path)

            # Adjust token length based on the model's vision tower
            if "qwen" not in args.model_path.lower():
                if "224" in cfg_pretrained.mm_vision_tower:
                    # suppose the length of text tokens is around 1000, from bo's report
                    least_token_number = args.for_get_frames_num * (16 // args.mm_spatial_pool_stride) ** 2 + 1000
                else:
                    least_token_number = args.for_get_frames_num * (24 // args.mm_spatial_pool_stride) ** 2 + 1000

                scaling_factor = math.ceil(least_token_number / 4096)
                if scaling_factor >= 2:
                    if "vicuna" in cfg_pretrained._name_or_path.lower():
                        print(float(scaling_factor))
                        overwrite_config["rope_scaling"] = {"factor": float(scaling_factor), "type": "linear"}
                    overwrite_config["max_sequence_length"] = 4096 * scaling_factor
                    overwrite_config["tokenizer_model_max_length"] = 4096 * scaling_factor

            tokenizer, model, image_processor, context_len = load_pretrained_model(
                args.model_path,
                args.model_base,
                model_name,
                load_8bit=args.load_8bit,
                overwrite_config=overwrite_config
            )
        else:
            tokenizer, model, image_processor, context_len = load_pretrained_model(
                args.model_path,
                args.model_base,
                model_name
            )
    else:
        pass  # Handle GPT-4V initialization if needed

    # Handle model-specific configurations
    if "gpt4v" != args.model_path:
        if hasattr(model.config, "force_sample"):
            args.force_sample = model.config.force_sample
        else:
            args.force_sample = False

        if hasattr(model.config, "add_time_instruction"):
            args.add_time_instruction = model.config.add_time_instruction
        else:
            args.add_time_instruction = False

    # Create the output directory if it doesn't exist
    output_dir = os.path.dirname(args.output_results)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Load input JSON
    with open(args.input_json, 'r', encoding='utf-8') as f:
        input_data = json.load(f)

    # Open the output_results file in append mode
    with open(args.output_results, "a", encoding='utf-8') as ans_file:
        for entry in tqdm(input_data, desc="Processing entries"):
            video_name = entry.get("video_name")

            if not video_name:
                print(f"Invalid entry found and skipped: {entry}")
                continue

            # Find the video file
            video_path = find_video_file(args.video_folder, video_name)
            if not video_path:
                print(f"Video file for '{video_name}' not found in '{args.video_folder}'. Skipping.")
                continue

            sample_set = {
                "video_name": video_name,
                "pred": ""
            }

            # Load the video
            if "gpt4v" != args.model_path:
                video, frame_time, video_time = load_video(video_path, args)
                video = image_processor.preprocess(video, return_tensors="pt")["pixel_values"].half().cuda()
                video = [video]
            else:
                spare_frames, frame_time, video_time = load_video_base64(video_path)
                interval = int(len(spare_frames) / args.for_get_frames_num) if args.for_get_frames_num > 0 else 1

            # Prepare the prompt
            if "gpt4v" != args.model_path:
                qs = args.prompt
                if args.add_time_instruction:
                    time_instruction = f"The video lasts for {video_time:.2f} seconds, and {len(video[0])} frames are uniformly sampled from it. These frames are located at {frame_time}. Please answer the following questions related to this video."
                    qs = f'{time_instruction}\n{qs}'
                if model.config.mm_use_im_start_end:
                    qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + "\n" + qs
                else:
                    qs = DEFAULT_IMAGE_TOKEN + "\n" + qs

                conv = conv_templates[args.conv_mode].copy()
                conv.append_message(conv.roles[0], qs)
                conv.append_message(conv.roles[1], None)
                prompt = conv.get_prompt()

                input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).cuda()
                if tokenizer.pad_token_id is None:
                    if "qwen" in tokenizer.name_or_path.lower():
                        print("Setting pad token to bos token for qwen model.")
                        tokenizer.pad_token_id = 151643

                attention_masks = input_ids.ne(tokenizer.pad_token_id).long().cuda()

                stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
                keywords = [stop_str]
                stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

                cur_prompt = args.prompt
            else:
                prompt = args.prompt

            system_error = ""

            # Run inference
            if "gpt4v" != args.model_path:
                with torch.inference_mode():
                    if "mistral" not in cfg_pretrained._name_or_path.lower():
                        output_ids = model.generate(
                            inputs=input_ids,
                            images=video,
                            attention_mask=attention_masks,
                            modalities="video",
                            do_sample=False,
                            temperature=0.0,
                            max_new_tokens=1024,
                            top_p=0.1,
                            num_beams=1,
                            use_cache=True,
                            stopping_criteria=[stopping_criteria]
                        )
                    else:
                        output_ids = model.generate(
                            inputs=input_ids,
                            images=video,
                            attention_mask=attention_masks,
                            modalities="video",
                            do_sample=False,
                            temperature=0.0,
                            max_new_tokens=1024,
                            top_p=0.1,
                            num_beams=1,
                            use_cache=True
                        )
                outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
            else:
                openai.api_key = args.api_key  # Your API key here

                max_num_retries = 0
                retry = 5
                PROMPT_MESSAGES = [
                    {
                        "role": "user",
                        "content": [
                            f"These are frames from a video that I want to upload. Answer me one question of this video: {prompt}",
                            *map(lambda x: {"image": x, "resize": 336}, spare_frames[::interval]),
                        ],
                    },
                ]
                params = {
                    "model": "gpt-4-vision-preview",  # Replace with the correct model name if needed
                    "messages": PROMPT_MESSAGES,
                    "max_tokens": 1024,
                }
                success_flag = False
                while max_num_retries < retry:
                    try:
                        result = openai.ChatCompletion.create(**params)
                        outputs = result.choices[0].message.content
                        success_flag = True
                        break
                    except Exception as inst:
                        if hasattr(inst, 'error'):
                            if inst.error.code == 'rate_limit_exceeded':
                                if "TPM" in inst.error.message:
                                    time.sleep(30)
                                    max_num_retries += 1
                                    continue
                                else:
                                    print(f"Unhandled rate limit error: {inst.error.message}")
                                    break
                            elif inst.error.code == 'insufficient_quota':
                                print(f'insufficient_quota key')
                                exit()
                            elif inst.error.code == 'content_policy_violation':
                                print(f'content_policy_violation')
                                system_error = "content_policy_violation"
                                break
                            print('Find error message in response: ', str(inst.error.message), 'error code: ', str(inst.error.code))
                        else:
                            print(f"An unexpected error occurred: {str(inst)}")
                        max_num_retries += 1
                        time.sleep(5)  # Wait before retrying
                if not success_flag:
                    print(f'Calling OpenAI failed after retrying for {max_num_retries} times. Skipping this entry.')
                    continue

            # Post-process outputs
            if "gpt4v" != args.model_path:
                if "mistral" not in cfg_pretrained._name_or_path.lower():
                    if outputs.endswith(stop_str):
                        outputs = outputs[: -len(stop_str)]
                outputs = outputs.strip()

            print(f"Video Name: {video_name}")
            print(f"Response: {outputs}\n")

            # Handle GPT-4V specific errors
            if "gpt4v" == args.model_path:
                if system_error == 'content_policy_violation':
                    sample_set["pred"] = "Content policy violation."
                elif system_error == "":
                    sample_set["pred"] = outputs
                else:
                    print("An unexpected system error occurred.")
                    continue
            else:
                sample_set["pred"] = outputs

            # Write the result to output_results
            ans_file.write(json.dumps(sample_set, ensure_ascii=False) + "\n")
            ans_file.flush()

    print(f"Inference completed. Results saved to {args.output_results}")

if __name__ == "__main__":
    args = parse_args()
    run_inference(args)
