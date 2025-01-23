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

    # Define the command-line arguments
    parser.add_argument("--video_folder", help="Path to the directory containing video files.", default='')
    parser.add_argument("--input_json", help="Path to the input JSON file containing keys.", default='')
    parser.add_argument("--output_results", help="Path to the output JSON file for storing results.", default='')
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

    return parser.parse_args()

def load_json_keys(json_file):
    """Load keys from the input JSON file."""
    with open(json_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    return list(data.keys())

def match_videos(json_keys, video_folder):
    """Match JSON keys with video file names."""
    matched_videos = {}
    for key in json_keys:
        # Assuming video files have .mp4 extension
        video_path = os.path.join(video_folder, f"{key}.mp4")
        if os.path.exists(video_path):
            matched_videos[key] = video_path
    return matched_videos

def load_video(video_path, args):
    if args.for_get_frames_num == 0:
        return np.zeros((1, 336, 336, 3)), "0.00s", 0.0
    vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
    total_frame_num = len(vr)
    video_time = total_frame_num / vr.get_avg_fps()
    fps = round(vr.get_avg_fps())
    frame_idx = [i for i in range(0, len(vr), fps)]
    frame_time = [i/fps for i in frame_idx]
    if len(frame_idx) > args.for_get_frames_num or args.force_sample:
        sample_fps = args.for_get_frames_num
        uniform_sampled_frames = np.linspace(0, total_frame_num - 1, sample_fps, dtype=int)
        frame_idx = uniform_sampled_frames.tolist()
        frame_time = [i/vr.get_avg_fps() for i in frame_idx]
    frame_time_str = ",".join([f"{i:.2f}s" for i in frame_time])
    spare_frames = vr.get_batch(frame_idx).asnumpy()
    return spare_frames, frame_time_str, video_time

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
    return base64Frames

def run_inference(args):
    """
    Run inference on videos matched from input_json using the specified model.

    Args:
        args: Command-line arguments.
    """
    # Initialize the model
    if "gpt4v" != args.model_path:
        model_name = get_model_name_from_path(args.model_path)
        # Set model configuration parameters if they exist
        if args.overwrite:
            overwrite_config = {
                "mm_spatial_pool_mode": args.mm_spatial_pool_mode,
                "mm_spatial_pool_stride": args.mm_spatial_pool_stride,
                "mm_newline_position": args.mm_newline_position
            }

            cfg_pretrained = AutoConfig.from_pretrained(args.model_path)

            if "qwen" not in args.model_path.lower():
                if "224" in cfg_pretrained.mm_vision_tower:
                    least_token_number = args.for_get_frames_num * (16 // args.mm_spatial_pool_stride)**2 + 1000
                else:
                    least_token_number = args.for_get_frames_num * (24 // args.mm_spatial_pool_stride)**2 + 1000

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
        model = None  # Placeholder, actual GPT-4V model handling below

    # Update args based on model config if available
    if model and hasattr(model.config, "force_sample"):
        args.force_sample = model.config.force_sample
    else:
        args.force_sample = False

    if model and hasattr(model.config, "add_time_instruction"):
        args.add_time_instruction = model.config.add_time_instruction
    else:
        args.add_time_instruction = False

    # Load and match videos
    json_keys = load_json_keys(args.input_json)
    matched_videos = match_videos(json_keys, args.video_folder)

    if not matched_videos:
        print("No matching videos found.")
        return

    # Load existing results if output_results exists, else start with empty dict
    if os.path.exists(args.output_results):
        with open(args.output_results, "r", encoding="utf-8") as f:
            try:
                existing_results = json.load(f)
            except json.JSONDecodeError:
                print("Output file is not a valid JSON. Starting with an empty dictionary.")
                existing_results = {}
    else:
        existing_results = {}

    # Iterate over each matched video
    for video_key, video_path in tqdm(matched_videos.items(), desc="Processing videos"):
        # Skip if already processed
        if video_key in existing_results:
            print(f"Skipping already processed video: {video_key}")
            continue

        sample_set = {}
        question = args.prompt
        sample_set["Q"] = question
        sample_set["video_name"] = video_path

        # Check if the video exists
        if os.path.exists(video_path):
            if "gpt4v" != args.model_path:
                video, frame_time, video_time = load_video(video_path, args)
                video = image_processor.preprocess(video, return_tensors="pt")["pixel_values"].half().cuda()
                video = [video]
            else:
                spare_frames = load_video_base64(video_path)
                interval = max(1, len(spare_frames) // args.for_get_frames_num)
                selected_frames = spare_frames[::interval]

        system_error = ""

        # Prepare the prompt and inputs based on the model
        if "gpt4v" != args.model_path:
            qs = question
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

            cur_prompt = question
        else:
            prompt = question

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

            PROMPT_MESSAGES = [
                {
                    "role": "user",
                    "content": [
                        f"These are frames from a video that I want to upload. Answer me one question of this video: {prompt}",
                        *map(lambda x: {"image": x, "resize": 336}, selected_frames),
                    ],
                },
            ]
            params = {
                "model": "gpt-4-vision-preview",
                "messages": PROMPT_MESSAGES,
                "max_tokens": 1024,
            }
            success_flag = False
            max_num_retries = 0
            retry = 5
            while max_num_retries < retry:
                try:
                    result = openai.ChatCompletion.create(**params)
                    outputs = result.choices[0].message.content
                    success_flag = True
                    break
                except Exception as inst:
                    if hasattr(inst, 'error') and hasattr(inst.error, 'code'):
                        if inst.error.code == 'rate_limit_exceeded':
                            if "TPM" in inst.error.message:
                                print("Rate limit exceeded. Sleeping for 30 seconds.")
                                time.sleep(30)
                                max_num_retries += 1
                                continue
                            else:
                                raise inst
                        elif inst.error.code == 'insufficient_quota':
                            print('Insufficient quota for OpenAI API key.')
                            exit()
                        elif inst.error.code == 'content_policy_violation':
                            print('Content policy violation encountered.')
                            system_error = "content_policy_violation"
                            break
                        print('Error message:', str(inst.error.message), 'Error code:', str(inst.error.code))
                    else:
                        print('Unexpected error:', str(inst))
                    max_num_retries += 1
                    time.sleep(5)  # Wait before retrying
                    continue

            if not success_flag:
                print(f'Calling OpenAI failed after retrying for {max_num_retries} times. Skipping video {video_key}.')
                continue

            if system_error == 'content_policy_violation':
                continue

        # Post-process the output
        if "gpt4v" != args.model_path:
            if "mistral" not in cfg_pretrained._name_or_path.lower():
                if outputs.endswith(stop_str):
                    outputs = outputs[: -len(stop_str)]
            outputs = outputs.strip()

        # Prepare the result
        sample_set["pred"] = outputs

        # Update the existing results dictionary
        existing_results[video_key] = sample_set

        # Write the updated results to the output_results file
        with open(args.output_results, "w", encoding="utf-8") as ans_file:
            json.dump(existing_results, ans_file, ensure_ascii=False, indent=4)

        print(f"Processed video: {video_key}\nResponse: {outputs}\n")

    print(f"Inference completed. Results saved to {args.output_results}.")


if __name__ == "__main__":
    args = parse_args()
    run_inference(args)
