import json

from huggingface_hub import HfApi


def build_tree(file_paths):
    """
    将一维文件路径列表转换为嵌套字典树
    """
    tree = {}
    for path in file_paths:
        # 去除路径前后的空格或斜杠，按 '/' 切分
        parts = path.strip("/").split("/")
        current_layer = tree

        # 遍历路径中的每一个节点（除了最后一个文件名）
        for part in parts[:-1]:
            if part not in current_layer or current_layer[part] is None:
                current_layer[part] = {}
            current_layer = current_layer[part]

        # 最后一个节点是文件名，根据你的格式，将其值设为 None
        if parts:
            current_layer[parts[-1]] = None

    return tree


# 1. 初始化 API 并获取文件列表
api = HfApi()
REPO_ID = "cvml-nus/assembly101"

print("正在从 Hugging Face 获取文件列表，请稍候...")
try:
    all_files = api.list_repo_files(repo_id=REPO_ID, repo_type="dataset")
    print(f"成功获取文件树，共计 {len(all_files)} 个文件。")

    # 2. 转换成树状结构
    json_tree = build_tree(all_files)

    # 3. 保存为 JSON 文件（ensure_ascii=False 保证中文字符不转码，indent=2 让格式美观）
    output_filename = "assembly101_tree.json"
    with open(output_filename, "w", encoding="utf-8") as f:
        json.dump(json_tree, f, ensure_ascii=False, indent=2)

    print(f"文件树已成功保存至本地: {output_filename}")

except Exception as e:
    print(f"获取或保存失败: {e}")
