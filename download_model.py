# Copyright (c) 2019  PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"
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

import sys
import os

LOCAL_PATH = os.path.dirname(os.path.abspath(__file__))
TEST_PATH = os.path.join(LOCAL_PATH, "..")
sys.path.append(TEST_PATH)

from test_utils import download_file_and_uncompress

model_urls = {
    # ResNet系列
    "ResNet18_pretrained":
    "https://paddle-imagenet-models-name.bj.bcebos.com/ResNet18_pretrained.tar",
    "ResNet34_pretrained":
    "https://paddle-imagenet-models-name.bj.bcebos.com/ResNet34_pretrained.tar",
    "ResNet50_pretrained":
    "https://paddle-imagenet-models-name.bj.bcebos.com/ResNet50_pretrained.tar",
    "ResNet101_pretrained":
    "https://paddle-imagenet-models-name.bj.bcebos.com/ResNet101_pretrained.tar",
    "ResNet152_pretrained":
    "https://paddle-imagenet-models-name.bj.bcebos.com/ResNet152_pretrained.tar",
    # ResNet系列
    "ResNet50_vc_pretrained":
    "https://paddle-imagenet-models-name.bj.bcebos.com/ResNet50_vc_pretrained.tar",
    "ResNet18_vd_pretrained":
    "https://paddle-imagenet-models-name.bj.bcebos.com/ResNet18_vd_pretrained.tar",
    "ResNet34_vd_pretrained":
    "https://paddle-imagenet-models-name.bj.bcebos.com/ResNet34_vd_pretrained.tar",
    "ResNet50_vd_pretrained":
    "https://paddle-imagenet-models-name.bj.bcebos.com/ResNet50_vd_pretrained.tar",
    "ResNet50_vd_v2_pretrained":
    "https://paddle-imagenet-models-name.bj.bcebos.com/ResNet50_vd_v2_pretrained.tar",
    "ResNet101_vd_vd_pretrained":
    "https://paddle-imagenet-models-name.bj.bcebos.com/ResNet101_vd_pretrained.tar",
    "ResNet152_vd_vd_pretrained":
    "https://paddle-imagenet-models-name.bj.bcebos.com/ResNet152_vd_pretrained.tar",
    "ResNet200_vd_vd_pretrained":
    "https://paddle-imagenet-models-name.bj.bcebos.com/ResNet200_vd_pretrained.tar",
    "ResNet50_vd_ssld_pretrained":
    "https://paddle-imagenet-models-name.bj.bcebos.com/ResNet50_vd_ssld_pretrained.tar",
    "ResNet50_vd_ssld_v2_pretrained":
    "https://paddle-imagenet-models-name.bj.bcebos.com/ResNet50_vd_ssld_v2_pretrained.tar",
    "ResNet101_vd_ssld_pretrained":
    "https://paddle-imagenet-models-name.bj.bcebos.com/ResNet101_vd_ssld_pretrained.tar",
}

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("usage:\n  python download_model.py ${MODEL_NAME}")
        exit(1)

    model_name = sys.argv[1]
    if not model_name in model_urls.keys():
        print("Only support: \n  {}".format("\n  ".join(
            list(model_urls.keys()))))
        exit(1)

    url = model_urls[model_name]
    download_file_and_uncompress(
        url=url,
        savepath=LOCAL_PATH,
        extrapath=LOCAL_PATH,
        extraname=model_name)

    print("Pretrained Model download success!")