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
