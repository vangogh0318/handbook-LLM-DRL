# Copyright 2025 The HuggingFace Team. All rights reserved.
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

import warnings
from dataclasses import dataclass, field
from typing import Optional

from transformers import TrainingArguments
import trl


@dataclass
class GRPOConfig(trl.GRPOConfig):
    """
    """

    normalize_reward: bool = field(
        default=False,
        metadata={"help": "Whether to normalize rewards"},
    )

    normalize_advantage: bool = field(
        default=False,
        metadata={"help": "Whether to normalize advantages"},
    )
