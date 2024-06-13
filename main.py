from utils.util import calculate_psnr, calculate_ssim, calculate_mse, calculate_average_quality, calculate_lpips

if __name__ == "__main__":
    # 加载图像
    img_correct = 'data/IJRR/correct_frame_20.png'
    img_EVSNN = 'data/IJRR/EVSNN_index_20.bmp'
    img_EV2ID = 'data/IJRR/EV2ID_index_20.png'

    # 计算PSNR和SSIM
    psnr_EVSNN = calculate_psnr(img_correct, img_EVSNN)
    ssim_EVSNN = calculate_ssim(img_correct, img_EVSNN)
    mse_EVSNN = calculate_mse(img_correct, img_EVSNN)
    lpips_EVSNN = calculate_lpips(img_correct, img_EVSNN)

    psnr_EV2ID = calculate_psnr(img_correct, img_EV2ID)
    ssim_EV2ID = calculate_ssim(img_correct, img_EV2ID)
    mse_EV2ID = calculate_mse(img_correct, img_EV2ID)
    lpips_EV2ID = calculate_lpips(img_correct, img_EV2ID)

    print(f"PSNR of EVSNN: {psnr_EVSNN}")
    print(f"PSNR of EV2ID: {psnr_EV2ID}\n")

    print(f"SSIM of EVSNN: {ssim_EVSNN}")
    print(f"SSIM of EV2ID: {ssim_EV2ID}\n")

    print(f"MSE of EVSNN: {mse_EVSNN}")
    print(f"MSE of EV2ID: {mse_EV2ID}\n")

    print(f"LPIPS of EVSNN: {lpips_EVSNN}")
    print(f"LPIPS of EV2ID: {lpips_EV2ID}\n")

    EVSNN_image_list = []
    EV2ID_image_list = []
    correct_image_list = '/Volumes/CenJim/train data/dataset/IJRR/poster_6dof/Event Camera Dataset/images'
    EVSNN_image_list.append('/Volumes/CenJim/train data/result/IJRR poster_6dof on pretrained EVSNN_LIF 6.9  '
                            'fixed-duration 44ms begin-time 0.015952s  ')
    EV2ID_image_list.append('/Volumes/CenJim/train data/result/IJRR poster_6dof on pretrained EV2ID 6.9  '
                            'fixed-duration 44ms begin-time 0.015952s  ')
    EVSNN_avg_data = calculate_average_quality(EVSNN_image_list, correct_image_list, 5)
    EV2ID_avg_data = calculate_average_quality(EV2ID_image_list, correct_image_list, 1)

    print(f"PSNR of EVSNN average: {EVSNN_avg_data['psnr']}")
    print(f"PSNR of EV2ID average: {EV2ID_avg_data['psnr']}\n")

    print(f"SSIM of EVSNN average: {EVSNN_avg_data['ssim']}")
    print(f"SSIM of EV2ID average: {EV2ID_avg_data['ssim']}\n")

    print(f"MSE of EVSNN average: {EVSNN_avg_data['mse']}")
    print(f"MSE of EV2ID average: {EV2ID_avg_data['mse']}\n")

    print(f"LPIPS of EVSNN average: {EVSNN_avg_data['lpips']}")
    print(f"LPIPS of EV2ID average: {EV2ID_avg_data['lpips']}\n")
