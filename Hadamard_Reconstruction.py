from distutils.file_util import move_file
import multiprocessing
import os
from re import T
import time
import cv2
import numpy as np
import math
from tqdm import tqdm
from distutils.dir_util import copy_tree
from multiprocessing import Pool, Manager
import shutil


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


# 파일 복사 (멀티스레드 연산용 함수)
def copy_file(path):
    shutil.copy2(path[0], path[1])


# Mask의 CC number를 계산
def get_num_of_chunk(i, j):
    # 주어진 basis에 해당하는 mask 파일 이름
    # +와 -는 CC number가 동일하기 때문에 +만 불러와서 계산해도 상관이 없다!
    mask_name = str(i) + "_" + str(j) + "_+.png"

    # 패턴을 폴더에서 불러온다.
    mask = cv2.imread(mask_name, cv2.IMREAD_GRAYSCALE)

    x = 0; y = 0
    cc_order_x = 1; cc_order_y = 1
    terminated = False

    # x CC order 계산
    while x < mask.shape[0]-1:
        if(mask[x][y] != mask[x+1][y] and x < mask.shape[0]-1):
            cc_order_x += 1
        x += 1

    # y CC order 계산
    while y < mask.shape[1]-1:
        if(mask[x][y] != mask[x][y+1] and x < mask.shape[0]-1):
            cc_order_y += 1
        y += 1

    return cc_order_x*cc_order_y


# Reconstruction Coefficient 측정 (Single-pixel power 측정)
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

    return float(intensity_1-intensity_2) / float(image.shape[0] * image.shape[1])


# Sort mask in power order
def sort_masks_in_pow_order(image):
    size_x = image.shape[0]; size_y = image.shape[1]

    # Multithread support
    p = Pool()

    # Complete Hadamard reconstruction을 한다
    # Intensity 값을 이용해 mask를 정렬해야함!
    print("Intensity 분석 중...")
    t1 = time.time()
    reconstructed_img_hadamard = p.starmap(make_reconstruction_coeff, [(image, i, j) for i in range(0, size_x) for j in range(0, size_y)])
    p.close
    p.join

    # starmap의 결과는 1차원 배열로 나오기 때문에 2d array로 다시 변환해 준다.
    reconstructed_img_hadamard = np.reshape(reconstructed_img_hadamard, (-1, size_x))

    # reconstructed_img_hadamard를 pow에 따라 내림차순 정렬할 때의 index를 정의
    # ex) 256x256 array의 [131, 11] index에 해당하는 element가 4번째로 큰 값을 가질 경우 pow_sorted_idx[3] = [131, 11]이 저장된다.
    pow_sorted_idx = np.column_stack(np.unravel_index(np.argsort((-np.abs(reconstructed_img_hadamard)).ravel()),
                                                      reconstructed_img_hadamard.shape))

    # pow_sorted_idx matrix 정보를 /patterns/(이미지 크기) 폴더에 저장
    # 이후 pow order mode로 reconstruction할 때 불러와서 활용된다.
    script_dir = os.path.dirname(os.path.abspath(__file__))
    pattern_original_dir_path = script_dir + "/patterns/" + str(img_size_x) + "x" + str(img_size_y)
    np.save(pattern_original_dir_path + '/np_pow_sorted_idx', pow_sorted_idx)
    t2 = time.time()
    print("Intensity 분석 완료! 분석에 걸린 시간 : ", t2-t1)

    # /patterns/(이미지 크기)_pow_sorted 폴더 생성
    # Sorted mask pattern이 저장되는 폴더
    # 이 폴더에 이미지가 intensity가 큰 순서대로 번호를 부여받아 저장된다.
    pattern_pow_sorted_dir_path = script_dir + "/patterns/" + str(img_size_x) + "x" + str(img_size_y) + '_pow_sorted'
    os.makedirs(pattern_pow_sorted_dir_path, exist_ok=True)

    pow_sorted_idx = np.load('./np_pow_sorted_idx.npy')

    # 옮겨갈 파일의 새로운 이름을 pow_sorted_idx에 따라 숫자를 부여해 나타낸다.
    # ex) 5_+.png -> 5번째로 intensity가 크게 나오는 mask 중 +mask
    mask1_path_lst = [[pattern_original_dir_path + '/' + str(a[0]) + "_" + str(a[1]) + "_+.png",\
         pattern_pow_sorted_dir_path + '/' + str(i) + "_+.png"] for i, a in enumerate(pow_sorted_idx)]
    mask2_path_lst = [[pattern_original_dir_path + '/' + str(a[0]) + "_" + str(a[1]) + "_-.png",\
         pattern_pow_sorted_dir_path + '/' + str(i) + "_-.png"] for i, a in enumerate(pow_sorted_idx)]

    # Mask 파일을 복사하고 이름을 바꾸어 다른 폴더에 저장
    print("Sorted mask file 저장 중...")
    t1 = time.time()
    p.map(copy_file, mask1_path_lst)
    p.map(copy_file, mask2_path_lst)
    p.close
    p.join
    t2 = time.time()
    print("Sorted mask file 복사 완료! 파일 복사에 걸린 시간 : {time}", t2-t1)


def sort_masks_in_cc_order(mask_shape):
    size_x = mask_shape[0]; size_y = mask_shape[1]

    # Multithread support
    p = Pool()

    # Mask들의 CC order를 구한다
    # CC order 값을 이용해 mask를 정렬해야함!
    print("CC order 분석 중...")
    t1 = time.time()
    cc_orders_of_masks = p.starmap(get_num_of_chunk, [(i, j) for i in range(0, size_x) for j in range(0, size_y)])
    p.close
    p.join

    # starmap의 결과는 1차원 배열로 나오기 때문에 2d array로 다시 변환해 준다.
    cc_orders_of_masks = np.reshape(cc_orders_of_masks, (-1, size_x))

    # cc_orders_of_masks를 pow에 따라 내림차순 정렬할 때의 index를 정의
    # ex) 256x256 array의 [131, 11] index에 해당하는 element가 4번째로 큰 값을 가질 경우 CC_sorted_idx[3] = [131, 11]이 저장된다.
    CC_sorted_idx = np.column_stack(np.unravel_index(np.argsort((-np.abs(cc_orders_of_masks)).ravel()),
                                                      cc_orders_of_masks.shape))

    # CC_sorted_idx matrix 정보를 /patterns/(이미지 크기) 폴더에 저장
    # 이후 CC order mode로 reconstruction할 때 불러와서 활용된다.
    script_dir = os.path.dirname(os.path.abspath(__file__))
    pattern_original_dir_path = script_dir + "/patterns/" + str(img_size_x) + "x" + str(img_size_y)
    np.save(pattern_original_dir_path + '/np_CC_sorted_idx', CC_sorted_idx)
    t2 = time.time()
    print("CC order 분석 완료! 분석에 걸린 시간 : ", t2-t1)

    # /patterns/(이미지 크기)_CC_sorted 폴더 생성
    # Sorted mask pattern이 저장되는 폴더
    # 이 폴더에 이미지가 CC order가 큰 순서대로 번호를 부여받아 저장된다.
    pattern_CC_sorted_dir_path = script_dir + "/patterns/" + str(img_size_x) + "x" + str(img_size_y) + '_CC_sorted'
    os.makedirs(pattern_CC_sorted_dir_path, exist_ok=True)

    CC_sorted_idx = np.load('./np_CC_sorted_idx.npy')

    # 옮겨갈 파일의 새로운 이름을 pow_sorted_idx에 따라 숫자를 부여해 나타낸다.
    # ex) 5_+.png -> 5번째로 intensity가 크게 나오는 mask 중 +mask
    mask1_path_lst = [[pattern_original_dir_path + '/' + str(a[0]) + "_" + str(a[1]) + "_+.png",\
         pattern_CC_sorted_dir_path + '/' + str(i) + "_+.png"] for i, a in enumerate(CC_sorted_idx)]
    mask2_path_lst = [[pattern_original_dir_path + '/' + str(a[0]) + "_" + str(a[1]) + "_-.png",\
         pattern_CC_sorted_dir_path + '/' + str(i) + "_-.png"] for i, a in enumerate(CC_sorted_idx)]

    # Mask 파일을 복사하고 이름을 바꾸어 다른 폴더에 저장
    print("Sorted mask file 저장 중...")
    t1 = time.time()
    p.map(copy_file, mask1_path_lst)
    p.map(copy_file, mask2_path_lst)
    p.close
    p.join
    t2 = time.time()
    print("Sorted mask file 복사 완료! 파일 복사에 걸린 시간 : {time}", t2-t1)



# 2D Hadamard reconstruction
def hadamard_complete_reconstruction(image):
    size_x = image.shape[0]; size_y = image.shape[1]
    reconstructed_img_hadamard = np.zeros([size_x, size_y])
    
    # Multithread support
    p = Pool()

    # Hadamard Reconstruction
    print("Reconstruction 중...")
    t1 = time.time()
    reconstructed_img_hadamard = p.starmap(make_reconstruction_coeff, [(image, i, j) for i in range(0, size_x) for j in range(0, size_y)])
    p.close
    p.join

    # starmap의 결과는 1차원 배열로 나오기 때문에 2d array로 다시 변환해 준다.
    reconstructed_img_hadamard = np.reshape(reconstructed_img_hadamard, (-1, size_x))

    # 이 결과에 Inverse Hadamard transform을 하여 복원된 이미지를 얻는다.
    reconstructed_img = np.array(hadamard_2d_transform_inv(reconstructed_img_hadamard), dtype=np.uint8)
    t2 = time.time()
    print("Reconstruction 완료! Reconstruction에 걸린 시간 : ", t2-t1)

    return reconstructed_img


# Power ordering을 이용해 Masking pattern의 일부만을 사용하여 이미지를 복원하는 함수
def hadamard_reconstruction_pow_order_frac(image, frac):
    size_x = image.shape[0]; size_y = image.shape[1]
    reconstructed_img_hadamard = np.zeros([size_x, size_y])

    # power sorted index 정보가 들어있는 numpy를 로드한다.
    # 해당 파일이 존재하지 않을 경우 오류를 내므로 주의
    pow_sorted_idx = np.load('./np_pow_sorted_idx.npy')
    sampling_max = int(frac*size_x*size_y)
    
    # Multithread support
    p = Pool()
    print("Reconstruction 중...")
    t1 = time.time()
    tmp = p.starmap(make_reconstruction_coeff, [(image, pow_sorted_idx[i][0], pow_sorted_idx[i][1]) for i in range(0, sampling_max)])
    p.close
    p.join

    # 이 경우 Masking pattern의 일부만을 사용해 이미지를 복원하기 때문에 starmap의 결과는 일부만 있는 상태이다.
    # 그러므로 사용한 mask에 해당하는 index에는 starmap의 결과를 채워주고, 아닌 경우엔 0인 상태로 놔둔다.
    for i in range(0, sampling_max):
        reconstructed_img_hadamard[pow_sorted_idx[i][0]][pow_sorted_idx[i][1]] = tmp[i]

    # starmap의 결과는 1차원 배열로 나오기 때문에 2d array로 다시 변환해 준다.
    reconstructed_img_hadamard = np.reshape(reconstructed_img_hadamard, (-1, size_x))

    # 이 결과에 Inverse Hadamard transform을 하여 복원된 이미지를 얻는다.
    reconstructed_img = np.array(hadamard_2d_transform_inv(reconstructed_img_hadamard), dtype=np.uint8)
    t2 = time.time()
    print("Reconstruction 완료! Reconstruction에 걸린 시간 : ", t2-t1)
    return reconstructed_img

if __name__=="__main__":

    # 이미지 읽어오기
    script_path = os.path.dirname(os.path.abspath(__file__))
    img_arr = np.fromfile(script_path+'\16x16.png', np.uint8)
    img = cv2.imdecode(img_arr, cv2.IMREAD_GRAYSCALE)
    img_size_x = img.shape[0]
    img_size_y = img.shape[1]

    
    terminated = False
    while not terminated:
        # 이미지 사이즈가 잘못되었을 경우 프로그램을 종료한다.
        if not float.is_integer(math.log2(img_size_x)) or not float.is_integer(math.log2(img_size_y)):
            print("Wrong Image size")
            input("종료하려면 아무 키나 누르세요....")
            break

        # Script가 있는 위치에 패턴 폴더가 존재하지 않는 경우 프로그램을 종료한다.
        pattern_path = script_path + "/patterns/" + str(img_size_x) + "x" + str(img_size_y)
        if not os.path.isdir(pattern_path):
            print("pattern folder not exists")
            input("종료하려면 아무 키나 누르세요....")
            break

        # 현재 작업 위치를 masking pattern이 들어있는 폴더로 바꾼다.
        os.chdir(pattern_path)

        # 작업 선택
        print("할 작업을 고르세요")
        print("1 : Sort masks in power order\n2 : Sort masks in CC order\nother : Reconstruct Image")
        usr_input = input("숫자를 입력하세요: ")
        print("\n\n")
        if int(usr_input) == 1:
            sort_masks_in_pow_order(img)
        elif int(usr_input) == 2:
            print("CC 추가 예정")
        else:
            print("1 : Power order reconstruction\n2 : CC order reconstruction\nother : Complete reconstruction")
            print("\n주의!: power order reconstruction의 경우 해당 이미지에 대한 power order가 저장된 numpy파일이 있어야 합니다.\
                저장된 numpy파일이 없을 경우 Sort masks in power order를 실행해 파일을 생성해주세요.")
            # 이미지 reconstruction
            usr_input = input("숫자를 입력하세요: ")
            if int(usr_input) == 1:
                usr_input = input("Sampling ratio를 설정하세요: ")
                if float(usr_input) <= 0 or float(usr_input) > 1 :
                    print("Wrong Input")
                    break
                img_reconstructed = hadamard_reconstruction_pow_order_frac(img, float(usr_input))
                terminated = True
                
            elif int(usr_input) == 2:
                print("CC 추가 예정")

            else:
                img_reconstructed = hadamard_complete_reconstruction(img)
                terminated = True
            
            # 다시 원래 작업 위치로 돌아와 파일 저장
            os.chdir(script_path)
            cv2.imshow("img", img)
            cv2.imshow("reconstructed img", img_reconstructed)
            cv2.imwrite("reconstructed_img.png", img_reconstructed)
            cv2.waitKey
            cv2.destroyAllWindows

                        