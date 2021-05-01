file_name = "m2.txt"
file_pointer = open(file_name)
lines = file_pointer.readlines()[1:]

pre_pro = []
for i in lines:
    pre_pro.append(i.split()[2:])

pre_rank = None
for i in range(len(pre_pro)):
    if len(pre_pro[i]) == 3:
        # rank만 있는 경우
        if 'R' == pre_pro[i][2][0]:
            pre_rank = pre_pro[i][2]
            pre_pro[i] = []
        # CQI 만 있는 경우
        else:
            if pre_rank == None:
                pre_pro[i] = []
            else:
                pre_pro[i].append(pre_rank)
        # 둘다
        # 있는 경우 rank만 갱신
    else:
        pre_rank = pre_pro[i][3]


while True:
    try:
        pre_pro.remove([])
    except ValueError:
        break


for i in range(0, len(pre_pro)):
    pre_pro[i][0] = int(pre_pro[i][0])

add_count = 0
for i in range(1, len(pre_pro)):
    if pre_pro[i][0] < pre_pro[i-1][0]-add_count:
        add_count += 1024
    pre_pro[i][0] += add_count
for i in pre_pro:
    print(i)