# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import os
import re
from io import BytesIO

import torch
from PIL import Image

from verl.utils.dataset.rl_dataset import RLHFDataset
from verl.utils.model import compute_position_id_with_mask

logger = logging.getLogger(__name__)

DEFAULT_SYSTEM_CONTENT = "You are a helpful and harmless assistant."
DEFAULT_USER_CONTENT_PREFIX = (
    "Answer the given question. You must conduct reasoning inside <reason> and </reason> "
    "first every time you get new information. After reasoning, if you find you lack "
    "some knowledge, you can call a search engine by <search_call> {\"name\":\"search\",\"arguments\":{\"query_list\":[\"your query\"]}} </search_call> "
    "and it will return the top searched results between <search_response> and "
    "</search_response>. You can search as many times as your want. If you find no "
    "further external knowledge needed, you can directly provide the answer inside "
    "<answer> and </answer>, without detailed illustrations. For example, "
    "<answer> Beijing </answer>. Question: "
)

class InfoseekRLHFDataset(RLHFDataset):
    """
    Custom dataset for Infoseek that handles local image paths without file:// prefix.
    """

    def _load_image_from_path(self, image_path: str) -> Image.Image:
        """Load image from local path string."""
        if isinstance(image_path, str):
            # Handle local path (with or without file:// prefix)
            if image_path.startswith("file://"):
                image_path = image_path[7:]  # Remove file:// prefix
            
            if os.path.exists(image_path):
                return Image.open(image_path).convert("RGB")
            else:
                logger.warning(f"Image not found: {image_path}")
                return None

    def _build_messages(self, example: dict):
        """Replace <image> and <video> placeholder in messages with corresponding image and video
        which is required by processor.apply_chat_template.
        - <image>: {"type": "image", "image": image}
        - <video>: {"type": "video", "video": video}

        Args:
            example: Row dictionary from dataframe.

        Returns:
            messages: List of messages with replaced placeholder.
        """
        messages: list = example[self.prompt_key]
        messages = [
            {
                "role": "system",
                "content": DEFAULT_SYSTEM_CONTENT
            },
            {
                "role": "user",
                "content": DEFAULT_USER_CONTENT_PREFIX + example[self.prompt_key][0]["content"]
            }
        ]
        images = example.pop(self.image_key, [])
        videos = example.pop(self.video_key, [])

        image_offset, video_offset = 0, 0
        for message in messages:
            if not images and not videos:
                continue
            assert self.processor is not None, "processor is needed to process image and video"

            content = message["content"]
            if not isinstance(content, str):
                continue

            content_list = []
            segments = re.split("(<image>|<video>)", content)
            segments = [item for item in segments if item != ""]
            for segment in segments:

                if segment == "<image>":
                    assert image_offset < len(images), f"image_offset {image_offset} >= len(images) {len(images)}"
                    image = images[image_offset]
                    if isinstance(image, Image.Image):
                        image = image.convert("RGB")
                    elif isinstance(image, dict) and "bytes" in image:
                        image["image"] = Image.open(BytesIO(image["bytes"]))
                    elif isinstance(image, str):
                        image = self._load_image_from_path(image)
                    else:
                        raise ValueError(f"Invalid image type: {type(image)}")

                    content_list.append({"type": "image", "image": image})
                    image_offset += 1
                elif segment == "<video>":
                    assert video_offset < len(videos), f"video_offset {video_offset} >= len(videos) {len(videos)}"
                    content_list.append({"type": "video", "video": videos[video_offset]})
                    video_offset += 1
                else:
                    content_list.append({"type": "text", "text": segment})
            message["content"] = content_list

        assert image_offset == len(images), f"image_offset {image_offset} != len(images) {len(images)}"
        assert video_offset == len(videos), f"video_offset {video_offset} != len(videos) {len(videos)}"
        return messages


    def __getitem__(self, item):
        """For rollout, apply_chat_template has been moved to AgentLoop, so we only return raw_prompt here."""
        row_dict: dict = self.dataframe[item]
        row_dict["raw_prompt"] = self._build_messages(row_dict)
        # TODO(wuxibin): We still need a dummy tensor to make sure DataProto.batch is not empty.
        # Remove this after deprecate DataProto by TensorDict.
        row_dict["dummy_tensor"] = torch.tensor([0], dtype=torch.uint8)

        # add index for each prompt
        if "extra_info" not in row_dict or row_dict["extra_info"] is None:
            row_dict["extra_info"] = dict()
        index = row_dict.get("extra_info", {}).get("index", 0)
        tools_kwargs = row_dict.get("extra_info", {}).get("tools_kwargs", {})
        interaction_kwargs = row_dict.get("extra_info", {}).get("interaction_kwargs", {})
        need_tools_kwargs = row_dict.get("extra_info", {}).get("need_tools_kwargs", self.need_tools_kwargs)
        if need_tools_kwargs and not tools_kwargs:
            logger.warning("tools_kwargs is empty for index {}, data source: {}", index, row_dict["data_source"])
        row_dict["index"] = index
        row_dict["tools_kwargs"] = tools_kwargs
        row_dict["interaction_kwargs"] = interaction_kwargs
        row_dict["agent_name"] = "tool_agent"

        return row_dict