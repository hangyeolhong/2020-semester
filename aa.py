import numpy as np

# x=np.array([[ 0.02291667,-0.02291667,0.02291667],
#  [ 0.02083333,-0.02083333,0.02083333],
#  [-0.02083333 , 0.02083333 , 0.02083333]])
# yy= np.array([
#     [1,2,3],[1,2,3],[3,4,4]
# ])
#
# # aa = np.array(yy).T[1].reshape(3, 1)
# # print(x.shape)
# # print(aa)
# # print(x*aa)
# print(np.sum(yy,axis=1))
# # print(np.sum(x*aa,axis=0))
#
# # print(x.T[2])
#
# aaa = np.array([1,2,3])
# aaa=aaa.reshape(1,aaa.size)
# print(aaa.ndim)
# print(aaa.shape)
#
# """
# def _numerical_gradient_no_batch_(f, x):
#     # returns gradient value of f at x
#     # x is a vector containing input value
#     h = 1e-4
#     grad = np.zeros_like(x)
#
#     for i in range(x.size):
#         temp = x[i]
#         x[i] = temp + h
#         fxh1 = f(x)
#
#         x[i] = temp - h
#         fxh2 = f(x)
#
#         grad[i] = (fxh1 - fxh2) / (2 * h)
#         x[i] = temp
#
#     return grad
#
#
# def numerical_gradient(f, x):
#     # x가 1차원이면 x의 각 원소에서의 f의 미분값을 리턴한다.
#     if x.ndim == 1:
#         return _numerical_gradient_no_batch_(f, x)
#
#     else:
#         grad = np.zeros_like(x)  # grad는 x와 형상이 같다.
#
#         # x의 각 행에 대하여
#         for i in range(x.shape[0]):
#             grad[i] = _numerical_gradient_no_batch_(f, x[i])  # 행의 원소에서의 f의 미분값을 구하고 grad의 ith 행에 저장
#
#         return grad
# """
# batch_mask = np.random.choice(120, 100)
# print(batch_mask)
y=np.array([[5,2],[3,9]])
batch_size=y.shape[0]
t=np.array([[0,1],[1,0]])
# t = t.argmax(axis=1)
# print(t)
# print(y[np.arange(batch_size), t])
# print( -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size)
print(y*t)
print(np.sum(y*t))