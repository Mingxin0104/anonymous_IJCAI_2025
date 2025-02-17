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
import pandas as pd  # 添加 pandas 用于读取 CSV 文件
import transformers
import torch
import argparse
import numpy as np
import pandas as pd
import os
import json
from tqdm import tqdm
from decord import VideoReader, cpu
import cv2


def parse_args():
    """
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser()
    # 定义命令行参数
    parser.add_argument("--video_path", help="Path to the video files.", default="")
    parser.add_argument("--video_folder", help="Path to the folder containing video files.", default='')
    parser.add_argument("--input_csv", help="Path to the input CSV file.", default='')
    parser.add_argument("--output_results", help="Path to the output results JSON file.", default='')
    parser.add_argument("--output_dir", help="Directory to save the model results JSON.", default="")
    parser.add_argument("--model_path", type=str, default="")
    parser.add_argument("--model_base", type=str, default=None)
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
    parser.add_argument("--load_8bit", type=lambda x: (str(x).lower() == 'true'), default=False)
    parser.add_argument("--prompt", type=str, default="Please provide a detailed description of the video, focusing on the main subjects, their actions, the background scenes.")
    parser.add_argument("--api_key", type=str, help="OpenAI API key")
    parser.add_argument("--mm_newline_position", type=str, default="grid")
    parser.add_argument("--force_sample", type=lambda x: (str(x).lower() == 'true'), default=False)
    parser.add_argument("--add_time_instruction", type=str, default=False)
    return parser.parse_args()


def load_video(video_path, args):
    if args.for_get_frames_num == 0:
        return np.zeros((1, 336, 336, 3))
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
    return base64Frames


def run_inference(args):
    # 从本地加载模型
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.model_path)
    model_config = transformers.AutoConfig.from_pretrained(args.model_path)
    # 确保 rope_scaling 只包含所需的字段
    if hasattr(model_config, "rope_scaling"):
        rope_scaling = model_config.rope_scaling
        if isinstance(rope_scaling, dict):
            correct_rope_scaling = {"type": rope_scaling.get("type"), "factor": rope_scaling.get("factor")}
            model_config.rope_scaling = correct_rope_scaling
    model = transformers.AutoModelForCausalLM.from_pretrained(args.model_path, torch_dtype=torch.bfloat16, config=model_config)
    pipeline = transformers.pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        model_kwargs={"torch_dtype": torch.bfloat16},
        device_map="auto",
    )
    # 创建输出目录（如果不存在）
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    output_results = args.output_results
    results_dict = {}
    # 如果输出文件已存在，加载已有的结果
    if os.path.exists(output_results):
        try:
            with open(output_results, "r", encoding="utf-8") as f:
                results_dict = json.load(f)
            print(f"已加载现有的结果文件：{output_results}")
        except json.JSONDecodeError:
            print(f"结果文件 {output_results} 无法解析为 JSON。将重新创建该文件。")
            results_dict = {}
    # 读取 CSV 文件
    try:
        df = pd.read_csv(args.input_csv)
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return
    if 'video' not in df.columns:
        print("CSV 文件中缺少 'video' 列。")
        return
    video_names_in_csv = [str(name).strip() + '.mp4' for name in df['video'].tolist()]
    video_folder_files = os.listdir(args.video_folder)
    video_folder_files_set = set(video_folder_files)
    for video_name in tqdm(video_names_in_csv, desc="Processing videos"):
        if video_name in video_folder_files_set:
            if video_name in results_dict:
                print(f"视频 '{video_name}' 已处理过，跳过。")
                continue
            video_path = os.path.join(args.video_folder, video_name)
            question = args.prompt
            if os.path.exists(video_path):
                video, frame_time, video_time = load_video(video_path, args)
            else:
                print(f"视频文件 '{video_path}' 不存在。跳过。")
                continue
            # 运行推理并获取输出
            qs = question
            if args.add_time_instruction:
                time_instruction = f"The video lasts for {video_time:.2f} seconds, and {len(video[0])} frames are uniformly sampled from it. These frames are located at {frame_time}. Please answer the following questions related to this video."
                qs = f'{time_instruction}\n{qs}'
            messages = [
                {"role": "system", "content": "You are an assistant that describes videos."},
                {"role": "user", "content": qs}
            ]
            outputs = pipeline(
                messages,
                max_new_tokens=256
            )
            generated_text = outputs[0]["generated_text"][-1]
            print(f"Question: {qs}\n")
            print(f"Response: {generated_text}\n")
            results_dict[video_name] = generated_text
            try:
                with open(output_results, "w", encoding="utf-8") as ans_file:
                    json.dump(results_dict, ans_file, ensure_ascii=False, indent=4)
                print(f"已更新结果到 {output_results}，视频名：{video_name}")
            except Exception as e:
                print(f"写入结果到 JSON 文件时出错：{e}")
        else:
            print(f"视频 '{video_name}' 不存在于 '{args.video_folder}' 中。跳过。")
    print(f"Inference completed. Results saved to {output_results}.")


if __name__ == "__main__":
    args = parse_args()
    run_inference(args)

















# qid

# import argparse
# import time
# import torch

# from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
# from llava.conversation import conv_templates, SeparatorStyle
# from llava.model.builder import load_pretrained_model
# from llava.utils import disable_torch_init
# from llava.mm_utils import process_anyres_image, tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria

# import json
# import os
# import math
# from tqdm import tqdm
# from decord import VideoReader, cpu

# from transformers import AutoConfig

# import cv2
# import base64
# import openai

# from PIL import Image

# import numpy as np
# import pandas as pd


# def split_list(lst, n):
#     """Split a list into n (roughly) equal-sized chunks"""
#     chunk_size = math.ceil(len(lst) / n)  # integer division
#     return [lst[i: i + chunk_size] for i in range(0, len(lst), chunk_size)]


# def get_chunk(lst, n, k):
#     chunks = split_list(lst, n)
#     return chunks[k]


# def parse_args():
#     """
#     Parse command-line arguments.
#     """
#     parser = argparse.ArgumentParser()

#     # 定义命令行参数
#     parser.add_argument("--video_path", help="Path to the video files.", default="/sda/wangwenhao/Code/LLaVA-NeXT/playground/demo/xU25MMA2N4aVtYay.mp4")
#     parser.add_argument("--video_folder", help="Path to the folder containing video files.", default='/sdb/ccfa/NExTVideo_unpack/')  # 新增参数
#     parser.add_argument("--input_csv", help="Path to the input CSV file.", default='/sdb/ccfa/code/test_next.csv')  # 新增参数
#     parser.add_argument("--output_results", help="Path to the output results JSON file.", default='/sda/wangwenhao/Code/LLaVA-NeXT/output_nextqa.json')  # 新增参数
#     parser.add_argument("--output_dir", help="Directory to save the model results JSON.", default="./work_dirs/video_demo/LLaVA-NeXT-Video-7B-Qwen2_vicuna_v1_frames_32_stride_2")
#     parser.add_argument("--model-path", type=str, default="/sdb/ccfa/video_understanding/LLaVA-NeXT-Video-7B-Qwen2/")
#     parser.add_argument("--model-base", type=str, default=None)
#     parser.add_argument("--conv-mode", type=str, default="vicuna_v1")
#     parser.add_argument("--chunk-idx", type=int, default=0)
#     parser.add_argument("--mm_resampler_type", type=str, default="spatial_pool")
#     parser.add_argument("--mm_spatial_pool_stride", type=int, default=2)
#     parser.add_argument("--mm_spatial_pool_out_channels", type=int, default=1024)
#     parser.add_argument("--mm_spatial_pool_mode", type=str, default="average")
#     parser.add_argument("--image_aspect_ratio", type=str, default="anyres")
#     parser.add_argument("--image_grid_pinpoints", type=str, default="[(224, 448), (224, 672), (224, 896), (448, 448), (448, 224), (672, 224), (896, 224)]")
#     parser.add_argument("--mm_patch_merge_type", type=str, default="spatial_unpad")
#     parser.add_argument("--overwrite", type=lambda x: (str(x).lower() == 'true'), default=True)
#     parser.add_argument("--for_get_frames_num", type=int, default=32)
#     parser.add_argument("--load_8bit", type=lambda x: (str(x).lower() == 'true'), default=False)
#     parser.add_argument("--prompt", type=str, default="Please provide a detailed description of the video, focusing on the main subjects, their actions, the background scenes.")
#     parser.add_argument("--api_key", type=str, help="OpenAI API key")
#     parser.add_argument("--mm_newline_position", type=str, default="grid")
#     parser.add_argument("--force_sample", type=lambda x: (str(x).lower() == 'true'), default=False)
#     parser.add_argument("--add_time_instruction", type=str, default=False)

#     return parser.parse_args()


# def load_video(video_path, args):
#     if args.for_get_frames_num == 0:
#         return np.zeros((1, 336, 336, 3))
#     vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
#     total_frame_num = len(vr)
#     video_time = total_frame_num / vr.get_avg_fps()
#     fps = round(vr.get_avg_fps())
#     frame_idx = [i for i in range(0, len(vr), fps)]
#     frame_time = [i / fps for i in frame_idx]
#     if len(frame_idx) > args.for_get_frames_num or args.force_sample:
#         sample_fps = args.for_get_frames_num
#         uniform_sampled_frames = np.linspace(0, total_frame_num - 1, sample_fps, dtype=int)
#         frame_idx = uniform_sampled_frames.tolist()
#         frame_time = [i / vr.get_avg_fps() for i in frame_idx]
#     frame_time = ",".join([f"{i:.2f}s" for i in frame_time])
#     spare_frames = vr.get_batch(frame_idx).asnumpy()
#     # import pdb;pdb.set_trace()

#     return spare_frames, frame_time, video_time


# def load_video_base64(path):
#     video = cv2.VideoCapture(path)

#     base64Frames = []
#     while video.isOpened():
#         success, frame = video.read()
#         if not success:
#             break
#         _, buffer = cv2.imencode(".jpg", frame)
#         base64Frames.append(base64.b64encode(buffer).decode("utf-8"))

#     video.release()
#     # print(len(base64Frames), "frames read.")
#     return base64Frames


# def run_inference(args):
#     """
#     使用 Video-ChatGPT 模型对 ActivityNet QA 数据集进行推理。

#     Args:
#         args: 命令行参数。
#     """
#     # 初始化模型
#     if "gpt4v"!= args.model_path:
#         model_name = get_model_name_from_path(args.model_path)
#         # 如果存在，设置模型配置参数
#         if args.overwrite:
#             overwrite_config = {}
#             overwrite_config["mm_spatial_pool_mode"] = args.mm_spatial_pool_mode
#             overwrite_config["mm_spatial_pool_stride"] = args.mm_spatial_pool_stride
#             overwrite_config["mm_newline_position"] = args.mm_newline_position

#             cfg_pretrained = AutoConfig.from_pretrained(args.model_path)

#             # import pdb;pdb.set_trace()
#             if "qwen" not in args.model_path.lower():
#                 if "224" in cfg_pretrained.mm_vision_tower:
#                     # 假设文本令牌的长度约为1000，根据bo的报告
#                     least_token_number = args.for_get_frames_num * (16 // args.mm_spatial_pool_stride) ** 2 + 1000
#                 else:
#                     least_token_number = args.for_get_frames_num * (24 // args.mm_spatial_pool_stride) ** 2 + 1000

#                 scaling_factor = math.ceil(least_token_number / 4096)
#                 if scaling_factor >= 2:
#                     if "vicuna" in cfg_pretrained._name_or_path.lower():
#                         print(float(scaling_factor))
#                         overwrite_config["rope_scaling"] = {"factor": float(scaling_factor), "type": "linear"}
#                     overwrite_config["max_sequence_length"] = 4096 * scaling_factor
#                     overwrite_config["tokenizer_model_max_length"] = 4096 * scaling_factor

#             tokenizer, model, image_processor, context_len = load_pretrained_model(
#                 args.model_path, args.model_base, model_name, load_8bit=args.load_8bit, overwrite_config=overwrite_config
#             )
#         else:
#             tokenizer, model, image_processor, context_len = load_pretrained_model(args.model_path, args.model_base, model_name)
#     else:
#         # 如果使用 GPT-4V，则不需要本地模型加载
#         model = None

#     # import pdb;pdb.set_trace()
#     if model and getattr(model.config, "force_sample", None) is not None:
#         args.force_sample = model.config.force_sample
#     else:
#         args.force_sample = False

#     # import pdb;pdb.set_trace()

#     if model and getattr(model.config, "add_time_instruction", None) is not None:
#         args.add_time_instruction = model.config.add_time_instruction
#     else:
#         args.add_time_instruction = False

#     # 创建输出目录（如果不存在）
#     if not os.path.exists(args.output_dir):
#         os.makedirs(args.output_dir)

#     output_results = args.output_results
#     results_dict = {}

#     # 如果输出文件已存在，加载已有的结果
#     if os.path.exists(output_results):
#         try:
#             with open(output_results, "r", encoding="utf-8") as f:
#                 results_dict = json.load(f)
#             print(f"已加载现有的结果文件：{output_results}")
#         except json.JSONDecodeError:
#             print(f"结果文件 {output_results} 无法解析为 JSON。将重新创建该文件。")
#             results_dict = {}

#     # 读取 CSV 文件
#     try:
#         df = pd.read_csv(args.input_csv)
#     except Exception as e:
#         print(f"Error reading CSV file: {e}")
#         return

#     if 'video_id' not in df.columns or 'qid' not in df.columns:  # 检查 qid 列是否存在
#         print("CSV 文件中缺少 'video_id' 或 'qid' 列。")
#         return

#     # 从 CSV 中提取视频名称，并添加 '.mp4' 扩展名
#     video_names_in_csv = [str(name).strip() + '.mp4' for name in df['video_id'].tolist()]

#     # 获取 video_folder 中所有视频文件的名称（包含扩展名）
#     video_folder_files = os.listdir(args.video_folder)
#     video_folder_files_set = set(video_folder_files)

#     # 遍历 CSV 中的每个视频名，检查是否在 video_folder 中存在匹配的视频
#     for video_name, qid in tqdm(zip(df['video_id'], df['qid']), desc="Processing videos"):  # 同时迭代 video_id 和 qid
#         full_video_name = str(video_name).strip() + '.mp4'
#         if full_video_name in video_folder_files_set:
#             if full_video_name in results_dict:
#                 print(f"视频 '{full_video_name}' 已处理过，跳过。")
#                 continue  # 跳过已处理过的视频

#             # 构造完整的视频路径
#             video_path = os.path.join(args.video_folder, full_video_name)

#             question = args.prompt

#             # 检查视频文件是否存在
#             if os.path.exists(video_path):
#                 if "gpt4v"!= args.model_path:
#                     video, frame_time, video_time = load_video(video_path, args)
#                     video = image_processor.preprocess(video, return_tensors="pt")["pixel_values"].half().cuda()
#                     video = [video]
#                 else:
#                     spare_frames, frame_time, video_time = load_video_base64(video_path)
#                     interval = int(len(spare_frames) / args.for_get_frames_num) if args.for_get_frames_num > 0 else 1
#             else:
#                 print(f"视频文件 '{video_path}' 不存在。跳过。")
#                 continue

#             # 运行推理并获取输出
#             if "gpt4v"!= args.model_path:
#                 qs = question
#                 if args.add_time_instruction:
#                     time_instruction = f"The video lasts for {video_time:.2f} seconds, and {len(video[0])} frames are uniformly sampled from it. These frames are located at {frame_time}. Please answer the following questions related to this video."
#                     qs = f'{time_instruction}\n{qs}'
#                 if model.config.mm_use_im_start_end:
#                     qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + "\n" + qs
#                 else:
#                     qs = DEFAULT_IMAGE_TOKEN + "\n" + qs

#                 conv = conv_templates[args.conv_mode].copy()
#                 conv.append_message(conv.roles[0], qs)
#                 conv.append_message(conv.roles[1], None)
#                 prompt = conv.get_prompt()

#                 input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).cuda()
#                 if tokenizer.pad_token_id is None:
#                     if "qwen" in tokenizer.name_or_path.lower():
#                         print("Setting pad token to bos token for qwen model.")
#                         tokenizer.pad_token_id = 151643

#                 attention_masks = input_ids.ne(tokenizer.pad_token_id).long().cuda()

#                 stop_str = conv.sep if conv.sep_style!= SeparatorStyle.TWO else conv.sep2
#                 keywords = [stop_str]
#                 stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

#                 cur_prompt = question
#             else:
#                 prompt = question

#             system_error = ""
#             outputs = ""

#             if "gpt4v"!= args.model_path:

#                 with torch.inference_mode():
#                     # model.update_prompt([[cur_prompt]])
#                     # import pdb;pdb.set_trace()
#                     # output_ids = model.generate(inputs=input_ids, images=video, attention_mask=attention_masks, modalities="video", do_sample=True, temperature=0.2, max_new_tokens=1024, use_cache=True, stopping_criteria=[stopping_criteria])
#                     if "mistral" not in cfg_pretrained._name_or_path.lower():
#                         output_ids = model.generate(
#                             inputs=input_ids,
#                             images=video,
#                             attention_mask=attention_masks,
#                             modalities="video",
#                             do_sample=False,
#                             temperature=0.0,
#                             max_new_tokens=1024,
#                             top_p=0.1,
#                             num_beams=1,
#                             use_cache=True,
#                             stopping_criteria=[stopping_criteria]
#                         )
#                         # output_ids = model.generate(inputs=input_ids, images=video, attention_mask=attention_masks, modalities="video", do_sample=True, temperature=0.2, max_new_tokens=1024, use_cache=True, stopping_criteria=[stopping_criteria])
#                     else:
#                         output_ids = model.generate(
#                             inputs=input_ids,
#                             images=video,
#                             attention_mask=attention_masks,
#                             modalities="video",
#                             do_sample=False,
#                             temperature=0.0,
#                             max_new_tokens=1024,
#                             top_p=0.1,
#                             num_beams=1,
#                             use_cache=True
#                         )
#                         # output_ids = model.generate(inputs=input_ids, images=video, attention_mask=attention_masks, modalities="video", do_sample=True, temperature=0.2, max_new_tokens=1024, use_cache=True)
#                 outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
#             else:
#                 openai.api_key = args.api_key  # 您的 API 密钥

#                 max_num_retries = 0
#                 retry = 5
#                 PROMPT_MESSAGES = [
#                     {
#                         "role": "user",
#                         "content": [
#                             f"These are frames from a video that I want to upload. Answer me one question of this video: {prompt}",
#                             *map(lambda x: {"image": x, "resize": 336}, spare_frames[0::interval]),
#                         ],
#                     },
#                 ]
#                 params = {
#                     "model": "gpt-4-vision-preview",  # gpt-4-1106-vision-preview
#                     "messages": PROMPT_MESSAGES,
#                     "max_tokens": 1024,
#                 }
#                 success_flag = False
#                 while max_num_retries < retry:
#                     try:
#                         result = openai.ChatCompletion.create(**params)
#                         outputs = result.choices[0].message.content
#                         success_flag = True
#                         break
#                     except Exception as inst:
#                         if 'error' in dir(inst):
#                             # import pdb;pdb.set_trace()
#                             if inst.error.code == 'rate_limit_exceeded':
#                                 if "TPM" in inst.error.message:
#                                     time.sleep(30)
#                                     max_num_retries += 1
#                                     continue
#                                 else:
#                                     import pdb
#                                     pdb.set_trace()
#                             elif inst.error.code == 'insufficient_quota':
#                                 print(f'insufficient_quota key')
#                                 exit()
#                             elif inst.error.code == 'content_policy_violation':
#                                 print(f'content_policy_violation')
#                                 system_error = "content_policy_violation"
#                                 break
#                             print('Find error message in response: ', str(inst.error.message), 'error code: ', str(inst.error.code))

#                         max_num_retries += 1
#                         continue
#                 if not success_flag:
#                     print(f'Calling OpenAI failed after retrying for {max_num_retries} times. Check the logs for details.')
#                     continue  # 跳过当前视频，继续下一个

#                 if "gpt4v" == args.model_path:
#                     if system_error == 'content_policy_violation':
#                         continue
#                     elif system_error == "":
#                         pass
#                     else:
#                         import pdb
#                         pdb.set_trace()

#             print(f"Question: {prompt}\n")
#             print(f"Response: {outputs}\n")

#             if "gpt4v"!= args.model_path:
#                 if "mistral" not in cfg_pretrained._name_or_path.lower():
#                     if outputs.endswith(stop_str):
#                         outputs = outputs[: -len(stop_str)]
#                 outputs = outputs.strip()

#             # 将结果添加到字典中，键名由 video_id 变为 video_id-qid
#             results_dict[f"{video_name}-{qid}"] = outputs  

#             # 将当前结果写入 JSON 文件
#             try:
#                 with open(output_results, "w", encoding="utf-8") as ans_file:
#                     json.dump(results_dict, ans_file, ensure_ascii=False, indent=4)
#                 print(f"已更新结果到 {output_results}，视频名：{video_name}-{qid}")
#             except Exception as e:
#                 print(f"写入结果到 JSON 文件时出错：{e}")

#         else:
#             print(f"视频 '{full_video_name}' 不存在于 '{args.video_folder}' 中。跳过。")

#     print(f"Inference completed. Results saved to {output_results}.")


# if __name__ == "__main__":
#     args = parse_args()
#     run_inference(args)
