import os
import random

# --- 설정 ---
# 디렉터리 이름의 접두사
prefix = "sequence_"
# 디렉터리 개수
num_sequences = 300
# 분할 비율 (train, val, test)
split_ratio = (0.8, 0.1, 0.1)

# --- 스크립트 시작 ---

# 1. 'sequence_001'부터 'sequence_300'까지의 디렉터리 이름 리스트 생성
all_ids = [f"{prefix}{str(i).zfill(3)}" for i in range(1, num_sequences + 1)]

# 2. 리스트를 무작위로 섞어 샘플링합니다.
random.shuffle(all_ids)

# 3. 분할 지점 계산
train_end = int(num_sequences * split_ratio[0])
val_end = train_end + int(num_sequences * split_ratio[1])

# 4. 리스트를 무작위 상태로 분할
train_list = all_ids[:train_end]
val_list = all_ids[train_end:val_end]
test_list = all_ids[val_end:]

# ###################### 핵심 수정 부분 ###################### #
# 5. 각 리스트를 파일에 저장하기 전에 오름차순으로 정렬합니다.
all_ids.sort()
train_list.sort()
val_list.sort()
test_list.sort()
# ######################################################### #

# 6. 정렬된 리스트를 파일로 저장
with open("all_ids.txt", "w") as f:
    for item in all_ids:
        f.write(f"{item}\n")

with open("train_list.txt", "w") as f:
    for item in train_list:
        f.write(f"{item}\n")

with open("val_list.txt", "w") as f:
    for item in val_list:
        f.write(f"{item}\n")

with open("test_list.txt", "w") as f:
    for item in test_list:
        f.write(f"{item}\n")

# 7. 결과 출력
print("파일 생성이 완료되었습니다. (내용 정렬됨) ✅")
print(f" - 총 {len(all_ids)}개")
print(f" - train_list.txt: {len(train_list)}개")
print(f" - val_list.txt: {len(val_list)}개")
print(f" - test_list.txt: {len(test_list)}개")