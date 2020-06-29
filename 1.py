import numpy as np

A= np.array([[1,0,0],[1,0,0],[1,1,1]])


sigma=np.array([[0.5,0],[0,1]])
b=np.array([[0],[2],[2]])
v1=np.array([[-0.816,-0.408,-0.408],[-0.577,0.577,0.577]])
u1h=np.array([[-0.408,-0.408,-0.817],[-0.577,-0.577,0.577]])

xmin=np.matmul(v1.T,(np.matmul(sigma,np.matmul(u1h,b))))

print(xmin)

AAT = np.dot(A,A.T)
ATA = np.matmul(A.T,A)
print(AAT)
print(ATA)

x1,v1 = np.linalg.eig(AAT)
x2,v2 = np.linalg.eig(ATA)

print(x1)
print(v1)
print(x2)
print(v2)


u,s,v=np.linalg.svd(A)
print(u)
print(s)
print(v)


sigma = np.array([[1/3,0],[0,1/2],[0,0]])
AA = np.matmul(v,np.matmul(sigma,u.T))
print(AA)
v4=np.array([[0.577,0.707,0],[0.577,0,0],[0.577,-0.707,0]])
AAA = np.matmul(v4,np.matmul(sigma,v2.T))
print(AAA)
v3=np.array([[0.707,0.707],[0.707,-0.707]])
AAAA = np.matmul(v1,np.matmul(sigma,v3.T))
print(AAAA)