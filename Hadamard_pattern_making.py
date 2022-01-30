import os
import cv2
import numpy as np
import math
import ffht
from tqdm import tqdm

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


# 2D 하다마드 변환
# return의 result 뒤에 붙는 pow는 hadamard transform의 coefficient
def hadamard_2d_transform(mat):
    result = ffht.fht(mat).transpose().copy()
    result = np.transpose(ffht.fht(result))
    return result * pow(2, -exp_x) * pow(2, -exp_y)


# 2D 하다마드 역변환
def hadamard_2d_transform_inv(mat):
    # 여기서 다루는 hadamard matrix 형태는 symmetric
    # 즉, Inverse hadamard matrix는 hadamard matrix와 같다.
    result = ffht.fht(mat).transpose().copy()
    result = np.transpose(ffht.fht(result))
    return result


# Make hadamard mask
def create_mask_img(size_x, size_y, i, j):

    # 이미지를 projection하는 hadamard mask pattern을 계산하여 구함
    delta_matrix = np.zeros([size_x, size_y])
    delta_matrix[i][j] = 1
    delta_hadamard_inv = hadamard_2d_transform_inv(delta_matrix)
    mask1 = 0.5 * (delta_hadamard_inv + 1)
    mask2 = 0.5 * (delta_hadamard_inv * (-1) + 1)
    mask1_img = np.array(mask1 * 255, dtype=np.uint8)
    mask2_img = np.array(mask2 * 255, dtype=np.uint8)
    mask1_name = str(i) + "_" + str(j) + "_+.png"
    mask2_name = str(i) + "_" + str(j) + "_-.png"
    # 이미지 저장
    cv2.imwrite(mask1_name, mask1_img)
    cv2.imwrite(mask2_name, mask2_img)


# 2D Hadamard reconstruction
def save_mask_imgs(size_x, size_y):
    reconstructed_img_hadamard = np.zeros([size_x, size_y])
    for i in tqdm(range(0, size_x)):
        for j in range(0, size_y):
            mask1_name = str(i) + "_" + str(j) + "_+.png"
            mask2_name = str(i) + "_" + str(j) + "_-.png"
            # 해당되는 Mask 패턴이 폴더에 없으면 패턴을 만든다.
            if not os.path.isfile(mask1_name) or not os.path.isfile(mask2_name):
                create_mask_img(size_x, size_y, i, j)


    return hadamard_2d_transform_inv(reconstructed_img_hadamard)


# 이미지 읽어오기
script_path = os.path.dirname(os.path.abspath(__file__))

terminated = False
while not terminated :
    img_size_y = int(input("만들고자 하는 Mask pattern의 x크기를 입력해주세요"))
    img_size_x = int(input("만들고자 하는 Mask pattern의 y크기를 입력해주세요"))
    if not float.is_integer(math.log2(img_size_x)) or not float.is_integer(math.log2(img_size_y)):
        print("Wrong Image size, image size should be powers of 2")
    else:
        terminated = True

# 현재 사진의 크기에 따라 pattern을 저장할 적절한 폴더를 만들어 작업 위치를 바꿈
pattern_path = script_path + "/patterns/" + str(img_size_x) + "x" + str(img_size_y)
os.makedirs(pattern_path, exist_ok=True)
os.chdir(pattern_path)

# mask image construction
save_mask_imgs(img_size_x, img_size_y)