from PIL import Image
import os
import torch
from torchvision import transforms
import shutil
import glob
from graph_classifier.src.model import ResNet50BinaryClassifier


# フィルタリングの閾値
MIN_IMAGE_AREA_RATIO = 0.0165
MIN_ASPECT_RATIO = 0.3
MAX_TABLE_RATIO = 0.27
TITLE_EXTENSION_RATIO = 0.15

def crop_figures_from_image(image_path: str, figure_positions: list, output_dir: str = "data/cropped_figures") -> list:
    """
    与えられた画像ファイルから、figure_positions の各ポリゴン領域をクロッピングし、
    フィルタリング基準を満たす場合にのみ保存します。

    Args:
        image_path (str): クロッピング元の画像ファイルパス
        figure_positions (list): 以下の形式のリスト
            [
              {'figure_index': 1, 'page_number': 1, 'polygon': [(x1, y1), (x2, y2), (x3, y3), (x4, y4)]},
              ...
            ]
        output_dir (str): クロッピング結果を保存するディレクトリ
    Returns:
        list: 保存したクロップ画像のファイルパスリスト
    """
    # 出力ディレクトリが存在しない場合は作成
    os.makedirs(output_dir, exist_ok=True)

    # 画像を開く
    img = Image.open(image_path)
    img_width, img_height = img.size  # 元画像のサイズを取得
    print(f"Processing image: {image_path} (Width: {img_width}, Height: {img_height})")
    cropped_files = []

    for fig in figure_positions:
        polygon = fig.get("polygon", [])
        if not polygon:
            continue

        # ポリゴンの各座標から、左上と右下の座標を算出（矩形領域に変換）
        xs = [p[0] for p in polygon]
        ys = [p[1] for p in polygon]
        left, upper, right, lower = min(xs), min(ys), max(xs), max(ys)
        cropped_width = right - left
        cropped_height = lower - upper
        cropped_area = cropped_width * cropped_height
        total_area = img_width * img_height
        aspect_ratio = cropped_width / max(cropped_height, 1)  # 高さがゼロになるのを防ぐ

        # **フィルタリングを適用（拡張前）**
        area_ratio = cropped_area / total_area
        if not (MIN_IMAGE_AREA_RATIO <= area_ratio <= MAX_TABLE_RATIO):
            print(f"❌ Figure {fig['figure_index']} skipped due to area ratio {area_ratio:.4f}")
            continue

        if aspect_ratio < MIN_ASPECT_RATIO:
            print(f"❌ Figure {fig['figure_index']} skipped due to low aspect ratio {aspect_ratio:.3f}")
            continue

        # **フィルタリングåを通過した場合のみ、タイトル部分を拡張**
        title_extension = int(cropped_height * TITLE_EXTENSION_RATIO)
        upper = max(0, upper - title_extension)  # y座標を上に拡張

        # **画像をクロッピング**
        bbox = (left, upper, right, lower)  # 修正後のバウンディングボックス
        cropped_img = img.crop(bbox)

        # **ファイル名を生成して保存**
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        save_path = os.path.join(output_dir, base_name)
        os.makedirs(save_path, exist_ok=True)
        output_file = os.path.join(save_path, f"figure_{fig['figure_index']}.png")
        cropped_img.save(output_file)
        cropped_files.append(output_file)
        print(f"✅ Cropped figure {fig['figure_index']} saved as {output_file}")

    return cropped_files


def is_graph_image(image_paths: list[str], 
                   device="cuda" if torch.cuda.is_available() else "cpu") -> list[str]:
    """
    指定された画像がグラフ・図かどうかを判定し、グラフと判定された画像のみをコピーする。

    Args:
        image_paths (List[str]): 判定したい画像のパスのリスト
        device (str): "cuda" または "cpu"

    Returns:
        List[str]: グラフと判定され、保存された画像のパスリスト
    """
    if not image_paths:
        return []

    model_path = "graph_classifier/model/model_best.pt"
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    # モデルのロード
    model = ResNet50BinaryClassifier()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    saved_files = []
    with torch.no_grad():
        for image_path in image_paths:
            image = Image.open(image_path).convert("RGB")
            image = transform(image).unsqueeze(0).to(device)  
            output = model(image).item()
            is_graph = output > 0.5

            if is_graph:
                new_path = image_path.replace("cropped_figures", "filtered_graphs")
                os.makedirs(os.path.dirname(new_path), exist_ok=True) 
                shutil.copy2(image_path, new_path)  
                saved_files.append(new_path)

    return saved_files

def get_cropped_image_paths(base_name: str, page_number: int, crop_dir: str = "data/cropped_figures") -> list[str]:
    """
    指定されたPDFのページに対応するクロップ済み画像のパスを取得する。

    Args:
        base_name (str): PDFのベース名（例: "1"）
        page_number (int): ページ番号（例: 2）
        crop_dir (str): クロップ済み画像の格納ディレクトリ（デフォルト: "data/cropped_figures"）

    Returns:
        List[str]: 指定されたページのクロップ済み画像のパスリスト（存在しない場合は空リスト）
    """
    # クロップ画像のディレクトリパスを作成（例: "data/cropped_figures/1_page_2/"）
    page_dir = os.path.join(crop_dir, f"{base_name}_page_{page_number}")

    # 画像が保存されているディレクトリが存在しない場合
    if not os.path.isdir(page_dir):
        return []

    # `figure_*.png` のパターンで全画像パスを取得
    image_paths = sorted(glob.glob(os.path.join(page_dir, "figure_*.png")))

    return image_paths