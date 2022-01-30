import os
import time
import cv2
import numpy as np
import math
from tqdm import tqdm
from distutils.dir_util import copy_tree

# num을 이진수로 표현했을 때 1의 개수 세기
def count_number_of_1(num):
    cnt = 0
    while num != 0:
        num &= (num - 1)
        cnt = cnt + 1
    return cnt


# 하다마드 행렬 H_n 생성
def hadamard_matrix(n):
    # Use a breakpoint in the code line below to debug your script.
    size = pow(2, n)
    hadamard = np.ones([size, size])
    for i in range(0, size):
        for j in range(0, size):
            if count_number_of_1(i & j) % 2 == 1:
                hadamard[i][j] = -1
    return hadamard


# 2D 하다마드 역변환
def hadamard_2d_transform_inv(mat):
    n_x = int(math.log2(mat.shape[0])); n_y = int(math.log2(mat.shape[1]))
    # 여기서 다루는 hadamard matrix 형태는 symmetric
    # 즉, Inverse hadamard matrix는 hadamard matrix와 같다.
    result = hadamard_matrix(n_x) @ mat
    result = result @ hadamard_matrix(n_y)
    return result


def make_reconstruction_coeff(image, i, j) :
            # 주어진 basis에 해당하는 mask 파일 이름
            mask1_name = str(i) + "_" + str(j) + "_+.png"
            mask2_name = str(i) + "_" + str(j) + "_-.png"

            # 패턴을 폴더에서 불러와 읽은 이미지에 적용한다.
            mask1 = cv2.imread(mask1_name, cv2.IMREAD_GRAYSCALE)
            mask2 = cv2.imread(mask2_name, cv2.IMREAD_GRAYSCALE)
            img_reflected_1 = np.array(image * (mask1 / 255), dtype=np.uint8)
            img_reflected_2 = np.array(image * (mask2 / 255), dtype=np.uint8)

            # 측정되는 intensity를 연산해 reconstructed image의 coefficient를 구한다.
            intensity_1 = np.sum(img_reflected_1.astype(dtype=np.int32), dtype=np.int32)
            intensity_2 = np.sum(img_reflected_2.astype(dtype=np.int32), dtype=np.int32)
            return intensity_1 - intensity_2


# Sort mask in power order
def sort_mask_in_pow_order(image):
    size_x = img.shape[0]
    size_y = img.shape[1]
    reconstructed_img_hadamard = np.zeros([size_x, size_y])

    for i in tqdm(range(0, size_x)):
        for j in range(0, size_y):
            reconstructed_img_hadamard[i][j] = float(make_reconstruction_coeff(image, i, j) / float(size_x * size_y))
    
    # reconstructed_img_hadamard를 pow에 따라 내림차순 정렬할 때의 index를 정의
    # ex) 256x256 array의 [131, 11] index에 해당하는 element가 4번째로 큰 값을 가질 경우 pow_sorted_idx[3] = [131, 11]이 저장된다.
    pow_sorted_idx = np.column_stack(np.unravel_index(np.argsort((-np.abs(reconstructed_img_hadamard)).ravel()),
                                                      reconstructed_img_hadamard.shape))
    # /patterns/(이미지 크기)_pow_sorted 폴더 생성
    # 이 폴더에 이미지가 intensity가 큰 순서대로 번호를 부여받아 저장된다.
    script_dir = os.path.dirname(os.path.abspath(__file__))
    par_script_dir = os.path.dirname(script_dir)
    pattern_original_dir_path = par_script_dir + "/" + str(img_size_x) + "x" + str(img_size_y)
    pattern_pow_sorted_dir_path = par_script_dir + '/' + str(img_size_x) + "x" + str(img_size_y) + '_pow_sorted'

    # pow_sorted_idx matrix 정보를 /patterns/(이미지 크기) 폴더에 저장
    # 이후 pow order mode로 reconstruction할 때 불러와서 활용된다.
    np.save('./np_pow_sorted_idx', pow_sorted_idx)
    copy_tree(pattern_original_dir_path, pattern_pow_sorted_dir_path)
    for i in tqdm(range(0, size_x*size_y)):
        mask1_name = script_dir+"/"+str(pow_sorted_idx[i][0]) + "_" + str(pow_sorted_idx[i][1]) + "_+.png"
        mask2_name = script_dir+"/"+str(pow_sorted_idx[i][0]) + "_" + str(pow_sorted_idx[i][1]) + "_+.png"
        os.rename(mask1_name, str(i) + "_+.png")
        os.rename(mask2_name, str(i) + "_-.png")


# 2D Hadamard reconstruction
def hadamard_reconstruction_complete(image):
    size_x = img.shape[0]; size_y = img.shape[1]
    reconstructed_img_hadamard = np.zeros([size_x, size_y])
    
    for i in tqdm(range(0, size_x)):
        for j in range(0, size_y):
            reconstructed_img_hadamard[i][j] = float(make_reconstruction_coeff(image, i, j)/float(size_x*size_y))

    reconstructed_img = np.array(hadamard_2d_transform_inv(reconstructed_img_hadamard), dtype=np.uint8)
    return reconstructed_img

def hadamard_reconstruction_pow_order_frac(image, frac):
    size_x = img.shape[0]; size_y = img.shape[1]
    reconstructed_img_hadamard = np.zeros([size_x, size_y])

    pow_sorted_idx = np.load('./np_pow_sorted_idx.npy').copy()
    sampling_max = int(frac*size_x*size_y)
    # idxs = [pow_sorted_idx[i] for i in range(0, sampling_max)]
    for i in range(0, sampling_max):
        idx_x = pow_sorted_idx[i][0]; idx_y = pow_sorted_idx[i][1]
        reconstructed_img_hadamard[idx_x][idx_y] = float(make_reconstruction_coeff(image, idx_x, idx_y) / float(size_x * size_y))

    reconstructed_img = np.array(hadamard_2d_transform_inv(reconstructed_img_hadamard), dtype=np.uint8)
    return reconstructed_img

# 이미지 읽어오기
script_path = os.path.dirname(os.path.abspath(__file__))
img_arr = np.fromfile(script_path+'\original.png', np.uint8)
img = cv2.imdecode(img_arr, cv2.IMREAD_GRAYSCALE)
img_size_x = img.shape[0]
img_size_y = img.shape[1]

if math.log2(img_size_x) % 2 != 0 or math.log2(img_size_y) % 2 != 0:
    print("Wrong Image size")

else:

    # 현재 사진의 크기에 따라 pattern을 저장할 적절한 폴더를 만들어 작업 위치를 바꿈
    pattern_path = script_path + "/patterns/" + str(img_size_x) + "x" + str(img_size_y)
    if os.path.isdir(pattern_path):
        print("Mask file detected")
        os.chdir(pattern_path)

        # 작업 선택
        print("할 작업을 고르세요")
        print("1 : Sort masks in power order\n2 : Sort masks in CC order\n3 : Reconstruct Image")
        usr_input = input("숫자를 입력하세요: ")
        if int(usr_input) == 1:
            sort_mask_in_pow_order(img)
        elif int(usr_input) == 2:
            print("CC 추가 예정")
        else:
            # 이미지 reconstruction
            img_reconstructed = hadamard_reconstruction_pow_order_frac(img, 0.35)
            # 다시 원래 작업 위치로 돌아와 파일 저장
            os.chdir(script_path)
            cv2.imshow("img", img)
            cv2.imshow("reconstructed img", img_reconstructed)
            cv2.imwrite("reconstructed_img.png", img_reconstructed)
            cv2.waitKey
            cv2.destroyAllWindows

    else:
        print("Pattern Folder Not Exists")