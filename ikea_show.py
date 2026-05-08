# datasets/ikea/annotations/gt_action.npy
import os

import cv2
import numpy as np
from decord import VideoReader, cpu, gpu
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

data_dir = r"/data/ssd2/thw/data/dataset/ikea"
path = os.path.join(data_dir, "annotations/gt_action.npy")
data = np.load(path, allow_pickle=True)
video_path = os.path.join(data_dir, "ANU_ikea_dataset_video")
target_fps = 4
vit = "google/siglip-large-patch16-384"
output_path = os.path.join(
    data_dir,
    f"ikea_action_dataset_{target_fps}fps_{vit.replace('/', '_').replace('-', '_')}",
)


def main():
    os.makedirs(output_path, exist_ok=True)
    # 先取出字典（.npy 存储的是 0-d object array，需要用 .item() 解包）
    data_dict = data.item()
    # 打印所有键
    print("所有键：", list(data_dict.keys()))

    scan_names = data_dict["scan_name"]  # 取 scan_name 字段
    gt_labels = data_dict["gt_labels"]
    model = AutoModel.from_pretrained(vit, device_map="cuda:0")
    tokenizer = AutoTokenizer.from_pretrained(vit)
    for name, label in tqdm(zip(scan_names, gt_labels)):
        print(label)
        print(name, label)
        video = os.path.join(video_path, name, "dev3/images/scan_video.avi")
        if not os.path.exists(video):
            print(f"Video not found: {video}")
            continue
        vr = VideoReader(video, ctx=cpu(0))
        # fps = vr.get_avg_fps()
        # # 计算索引：从 60fps 降低到 4fps，即每 15 帧取一帧
        # indices = []
        # length = min(len(vr), len(label))
        # new_labels = []
        # for i in range(length):
        #     if i % (fps // target_fps) == 0:
        #         indices.append(i)
        #         new_labels.append(label[i]) tt
        # # 一次性提取所有指定帧，返回的是 decord 数组，转成 numpy 极快
        # frames = vr.get_batch(indices).asnumpy()
        # new_labels = np.array(new_labels)
        # frames = frames.transpose(0, 3, 1, 2)
        # print(frames.shape, new_labels.shape)


if __name__ == "__main__":
    main()
