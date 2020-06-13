#-*-coding:utf-8-*-

filename = open('./0.25_CONSIST_SINGLE_L2.txt', 'r')
ground_truth = []
pre_result = []
great = []
i = 0

lines = filename.readlines()
for c in lines:
    c = c.rstrip("\n")
    d = c.split(", ")
    pre_result.append(d[1])
    #e = int(d[0]) + 1
    ground_truth.append(d[0])
    #ground_truth.append(str(e))

filename.close()
print(len(pre_result))
print(pre_result)
print(ground_truth)


while i < len(ground_truth):
    if ground_truth[i] == pre_result[i]:
        great.append(ground_truth[i])
    i = i+1

print(len(great))
acc = float(len(great))/len(ground_truth)
print(acc)
