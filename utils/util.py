from skimage.metrics import mean_squared_error as compare_mse
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim
import cv2
import os
import numpy as np


def histogram_normalization(img_dir, gray_flag: bool = True):
    # 如果图像是彩色图像，将其转换为YUV颜色空间
    if gray_flag:
        image = cv2.imread(img_dir, cv2.IMREAD_GRAYSCALE)
    else:
        image = cv2.imread(img_dir, cv2.IMREAD_COLOR)

    # 将图像转换为浮点数格式
    image_float = image.astype(np.float32)

    # 计算最小值和最大值
    min_val = np.min(image_float)
    max_val = np.max(image_float)

    # 归一化处理，将图像像素值缩放到0到255范围内
    normalized_image = 255 * (image_float - min_val) / (max_val - min_val)

    # 将图像转换回8位无符号整数格式
    normalized_image = normalized_image.astype(np.uint8)

    return normalized_image


def calculate_psnr(img_1_dir, img_2_dir, gray_flag: bool = True):
    img_correct = histogram_normalization(img_1_dir, gray_flag)
    img_compared = histogram_normalization(img_2_dir, gray_flag)
    # if gray_flag:
    #     img_correct = cv2.imread(img_1, cv2.IMREAD_GRAYSCALE)
    #     img_compared = cv2.imread(img_2, cv2.IMREAD_GRAYSCALE)
    # else:
    #     img_correct = cv2.imread(img_1, cv2.IMREAD_COLOR)
    #     img_compared = cv2.imread(img_2, cv2.IMREAD_COLOR)
    psnr = compare_psnr(img_correct, img_compared)
    return psnr


def calculate_ssim(img_1_dir, img_2_dir, gray_flag: bool = True):
    img_correct = histogram_normalization(img_1_dir, gray_flag)
    img_compared = histogram_normalization(img_2_dir, gray_flag)
    # if gray_flag:
    #     img_correct = cv2.imread(img_1, cv2.IMREAD_GRAYSCALE)
    #     img_compared = cv2.imread(img_2, cv2.IMREAD_GRAYSCALE)
    # else:
    #     img_correct = cv2.imread(img_1, cv2.IMREAD_COLOR)
    #     img_compared = cv2.imread(img_2, cv2.IMREAD_COLOR)
    ssim = compare_ssim(img_correct, img_compared)
    return ssim


def calculate_mse(img_1_dir, img_2_dir, gray_flag: bool = True):
    img_correct = histogram_normalization(img_1_dir, gray_flag)
    img_compared = histogram_normalization(img_2_dir, gray_flag)
    # if gray_flag:
    #     img_correct = cv2.imread(img_1, cv2.IMREAD_GRAYSCALE)
    #     img_compared = cv2.imread(img_2, cv2.IMREAD_GRAYSCALE)
    # else:
    #     img_correct = cv2.imread(img_1, cv2.IMREAD_COLOR)
    #     img_compared = cv2.imread(img_2, cv2.IMREAD_COLOR)
    mse = compare_mse(img_correct, img_compared)
    return mse


def calculate_average_quality(img_directory_list: list, correct_img_directory: str, step: int = 1):
    correct_files = [f for f in os.listdir(correct_img_directory) if
                     os.path.isfile(os.path.join(correct_img_directory, f)) and (
                             os.path.splitext(f)[1] == '.png' or os.path.splitext(f)[1] == '.bmp' or
                             os.path.splitext(f)[1] == '.jpg')]
    correct_files.sort()
    num_of_files = 0
    psnr_all = 0.0
    ssim_all = 0.0
    mse_all = 0.0
    for directory in img_directory_list:
        compare_files = [f for f in os.listdir(directory) if
                         os.path.isfile(os.path.join(directory, f)) and (
                             os.path.splitext(f)[1] == '.png' or os.path.splitext(f)[1] == '.bmp' or
                             os.path.splitext(f)[1] == '.jpg')]
        compare_files.sort()
        for index, filename in enumerate(correct_files):
            num_of_files = num_of_files + 1
            psnr_all = psnr_all + calculate_psnr(os.path.join(correct_img_directory, filename),
                                                 os.path.join(directory, compare_files[(index + 1) * step - 1]))
            ssim_all = ssim_all + calculate_ssim(os.path.join(correct_img_directory, filename),
                                                 os.path.join(directory, compare_files[(index + 1) * step - 1]))
            mse_all = mse_all + calculate_mse(os.path.join(correct_img_directory, filename),
                                              os.path.join(directory, compare_files[(index + 1) * step - 1]))
    return {'psnr': psnr_all / num_of_files, 'ssim': ssim_all / num_of_files, 'mse': mse_all / num_of_files}


def rename_files_in_directory(directory, prefix="file", add_old=False):
    """
        Change the filenames in the specified directory, generating filenames in Arabic numerical order.

        Parameters.
        directory: path of the directory where the filename should be changed (string).
        prefix: the prefix of the new filename (string), default is "file".
        add_old: whether add the old filename as the prefix ot not.
    """
    try:
        # obtain all the files under the directory
        files = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
        files.sort()

        # rename
        for index, filename in enumerate(files):
            # get the file extensions
            extension = os.path.splitext(filename)[1]
            if add_old:
                new_name = f"{filename.split('.')[0]}{prefix}{index}{extension}"
            else:
                new_name = f"{prefix}{index}{extension}"
            # generate full directory
            old_path = os.path.join(directory, filename)
            new_path = os.path.join(directory, new_name)
            os.rename(old_path, new_path)
            print(f"Renamed: {old_path} -> {new_path}")

        print("all files renamed")
    except Exception as e:
        print(f"error occur: {e}")


# 示例用法
if __name__ == "__main__":
    directory_path = '/Users/macbookpro/python_proj/vision_quality_compare/data/IJRR/correct_frame_20.png'
    cv2.imwrite('/Users/macbookpro/python_proj/vision_quality_compare/data/IJRR/correct_frame_20_normalized.png',
                histogram_normalization(directory_path))
