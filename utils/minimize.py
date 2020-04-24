import numpy as np
from scipy import optimize
import cv2

def loss_func(x):
    #### l2 loss
    return np.square(x), 2.0*x# do not use this.....
    #### robust l1 loss
    #tmp = np.sqrt(np.square(x)+1e-6)
    #return tmp, x/tmp
    #### cauchy loss
    # tmp = np.square(x)+1.
    # return np.log(tmp), 2.*x/tmp
    #### smooth l1 loss


def fun_and_grad(p, points, pd_planes, plist, pweight, olist, oweight, clist, cweight, colist, coweight, alpha):
    planes = p.reshape(-1,4)
    num_planes = planes.shape[0]
    fval = 0.
    grad_normal = np.zeros((num_planes,3))
    grad_offset = np.zeros(num_planes)
    # plane normal should fit PlaneRCNN 3D points
    aid = 0
    for i, xyz in enumerate(points):
        tmp = np.sum(planes[i:i+1,:3]*xyz, axis=1, keepdims=True) - planes[i,3]
        tmp_val, tmp_grad = loss_func(tmp)
        weight = alpha[aid]/num_planes/xyz.shape[0]
        fval += np.sum(tmp_val)*weight
        grad_normal[i,:] += np.sum(tmp_grad*xyz, axis=0)*weight
        grad_offset[i] += np.sum(-tmp_grad)*weight
    # normal close to planercnn
    aid += 1
    tmp = np.sum(planes[:,:3]*pd_planes[:,:3], axis=1, keepdims=True) - 1.0
    tmp_val, tmp_grad = loss_func(tmp)
    weight = alpha[aid]/num_planes
    fval += np.sum(tmp_val)*weight
    grad_normal += tmp_grad*pd_planes[:,:3]*weight

    # self normalization
    aid += 1
    tmp = np.sum(planes[:,:3]*planes[:,:3], axis=1, keepdims=True) - 1.0
    tmp_val, tmp_grad = loss_func(tmp)
    weight = alpha[aid]/num_planes
    fval += np.sum(tmp_val)*weight
    grad_normal += tmp_grad*planes[:,:3]*2.*weight

    # parallel list
    aid += 1 
    pnum = plist.shape[0]
    if pnum>0:
        lst1 = plist[:,0]
        lst2 = plist[:,1]
        tmp = np.sum(planes[lst1,:3]*planes[lst2,:3], axis=1, keepdims=True) - 1.0
        tmp_val, tmp_grad = loss_func(tmp)
        weight = alpha[aid]*pweight
        fval += np.sum(tmp_val*weight)
        for i in range(pnum):
            grad_normal[lst1[i],:] += tmp_grad[i]*planes[lst2[i],:3]*weight[i]
            grad_normal[lst2[i],:] += tmp_grad[i]*planes[lst1[i],:3]*weight[i]

    # orthogonal list
    aid += 1
    onum = olist.shape[0]
    if onum>0:
        lst1 = olist[:,0]
        lst2 = olist[:,1]
        tmp = np.sum(planes[lst1,:3]*planes[lst2,:3], axis=1, keepdims=True)
        tmp_val, tmp_grad = loss_func(tmp)
        weight = alpha[aid]*oweight
        fval += np.sum(tmp_val*weight)
        for i in range(onum):
            grad_normal[lst1[i],:] += tmp_grad[i]*planes[lst2[i],:3]*weight[i]
            grad_normal[lst2[i],:] += tmp_grad[i]*planes[lst1[i],:3]*weight[i]

    # contact
    aid += 1
    cnum = len(clist)
    if cnum>0:
        for i in range(cnum):
            pid1 = clist[i][0]
            pid2 = clist[i][1]
            plane1 = planes[pid1,:]
            plane2 = planes[pid2,:]
            raydirs = clist[i][2]
            pixnum = raydirs.shape[0]
            if pixnum == 0:
               continue
            tmp = plane2[3]*plane1[:3] - plane1[3]*plane2[:3]
            tmp = tmp.reshape(-1,3)
            tmp1 = np.sum(tmp*raydirs, axis=1, keepdims=True)
            tmp_val, tmp_grad = loss_func(tmp1)
            weight = alpha[aid]/pixnum*cweight[i]
            fval += np.sum(tmp_val)*weight
            for j in range(3):
               grad_normal[pid1,j] += np.sum(plane2[3]*raydirs[:,j]*tmp_grad.reshape(-1))*weight
               grad_normal[pid2,j] += np.sum(-plane1[3]*raydirs[:,j]*tmp_grad.reshape(-1))*weight
            grad_offset[pid1] += np.sum(-np.sum(plane2[:3].reshape(-1,3)*raydirs, axis=1, keepdims=True)*tmp_grad)*weight
            grad_offset[pid2] += np.sum(np.sum(plane1[:3].reshape(-1,3)*raydirs, axis=1, keepdims=True)*tmp_grad)*weight
    # coplane
    aid += 1
    conum = len(colist)
    if conum>0:
        lst1 = colist[:,0]
        lst2 = colist[:,1]
        tmp_val, tmp_grad = loss_func(planes[lst1,3:4] - planes[lst2,3:4])
        weight = alpha[aid]*coweight
        fval += np.sum(tmp_val*weight)
        for i in range(conum):
            grad_offset[lst1[i]] += tmp_grad[i]*weight[i]
            grad_offset[lst2[i]] -= tmp_grad[i]*weight[i]
    # put together
    grad = np.concatenate([grad_normal, grad_offset.reshape(-1,1)], axis=1).reshape(-1)
    return fval, grad

def validate_gradients(p, points, plist, pweight, olist, oweight, clist, cweight, colist, coweight, alpha):
# def validate_gradients():
#     p = np.random.randn(40)
#     points = []
#     for i in range(10):
#         j = np.random.randint(0,high=100)
#         points.append(np.random.randn(j,3))
#     plist = np.random.randint(0,high=10,size=(5,2))
#     olist = np.random.randint(0,high=10,size=(3,2))
#     alpha = np.array([1.,1.,1.,1.,1.])
#     clist = []
#     for i in range(7):
#         clist.append([np.random.randint(0,high=10), np.random.randint(0,high=10), np.random.random((np.random.randint(0,high=100),3))])
    p = np.random.randn(p.shape[0]*4)
    fval, grad = fun_and_grad(p, points, plist, pweight, olist, oweight, clist, cweight, colist, coweight, alpha)
    
    n_grad = np.zeros(grad.shape)
    h = 1e-7
    for i in range(grad.shape[0]):
        temp = p.copy()
        temp[i] += h
        fval1, _ = fun_and_grad(temp, points, plist, pweight, olist, oweight, clist, cweight, colist, coweight, alpha)
        n_grad[i] = (fval1 - fval)/h
    print(np.absolute(grad-n_grad))
    print(np.mean(np.absolute(grad-n_grad)))

def plane_minimize(pd_planes, pd_points, plist, pweight, olist, oweight, clist, cweight, colist, coweight, alpha):
    # initial guess
    p0 = pd_planes.copy().reshape(-1) # use planercnn as initial guess
    #p0=np.random.randn(pd_planes.shape[0]*pd_planes.shape[1])

    # bounds
    #bnds = ((-1,1),(-1,1),(-1,1),(None,None),)*pd_planes.shape[0]
    # start optimize
    #res = optimize.minimize(fun_and_grad, p0, args=(pd_points, plist, pweight, olist, oweight, clist, alpha), method='L-BFGS-B', jac=True, bounds=bnds)
    res = optimize.minimize(fun_and_grad, p0, args=(pd_points, pd_planes, plist, pweight, olist, oweight, clist, cweight, colist, coweight, alpha), method='BFGS', jac=True)
    re_planes = res.x
    re_planes = re_planes.reshape(-1,4)
    # normalize
    tmp = np.linalg.norm(re_planes[:,:3], axis=-1, keepdims=True)
    re_planes[:,:3] = re_planes[:,:3] / np.maximum(tmp, 1e-4)
    print(res.message, res.fun)
    return re_planes

def get_magnitude(image):

    gray = cv2.cvtColor(cv2.GaussianBlur(image,(5,5),0), cv2.COLOR_BGR2GRAY)
    sobelx = cv2.Sobel(gray,cv2.CV_64F,1,0,ksize=5)
    sobely = cv2.Sobel(gray,cv2.CV_64F,0,1,ksize=5)
    #print(np.amax(cv2.magnitude(sobelx, sobely)))
    return cv2.magnitude(sobelx, sobely)

def weighted_line_fitting_2d(x, y, weight):
    # ax+by+c=0 ransac
    npts = len(x)
    B = np.asarray([x, y, np.ones_like(x)])
    min_error = np.inf
    for iter in range(min(1000,5*npts)):
        sample_id = np.random.randint(npts, size=4)
        x_sample = x[sample_id]
        y_sample = y[sample_id]
        A = np.asarray([x_sample, y_sample, np.ones_like(x_sample)])
        e_vals, e_vecs = np.linalg.eig(np.matmul(A,A.T))
        abc = e_vecs[:, np.argmin(e_vals)]

        rest = np.absolute(np.matmul(abc.reshape(1,3), B).reshape(-1))*weight
        error = np.sum(rest)
        if error < min_error:
            min_error = error
            min_abc = abc.copy()

    rest = np.matmul(min_abc.reshape(1,3), B).reshape(-1)
    flag = (np.absolute(rest)<0.001)
    return flag
    


def toy_fun_and_grad(p, x, y):
    fval = np.square(x*p[0] + p[1] - y)
    fval = np.sum(fval)

    grad_0 = np.sum(x*(x*p[0] + p[1] - y))*2.0
    grad_1 = np.sum(x*p[0] + p[1] - y)*2.0
    return fval, np.array([grad_0, grad_1])

def toy_minimize():
    x = np.random.randn(10)
    p = np.array([-3,5])
    y = x*p[0] + p[1] + np.random.random(10)*0.1

    p0 = np.zeros(2)

    res = optimize.minimize(toy_fun_and_grad, p0, args=(x, y), method='BFGS', jac=True)
    print(res)

def fun_and_grad1111(p, pd_planes, plist, olist, clist, alpha):
    num_planes = pd_planes.shape[0]
    planes = p.reshape(-1,4)
    fval = 0.
    grad_normal = np.zeros((planes.shape[0],3))
    grad_offset = np.zeros(planes.shape[0])
    # plane normal close to planercnn
    tmp = np.sum(planes[:,:3]*pd_planes[:,:3], axis=1, keepdims=True) - 1.0
    tmp_val, tmp_grad = loss_func(tmp)
    weight = alpha[0]/num_planes
    fval += np.sum(tmp_val)*weight
    grad_normal += tmp_grad*pd_planes[:,:3]*weight
    # plane offset close to planercnn
    tmp_val, tmp_grad = loss_func(planes[:,3] - pd_planes[:,3])
    weight = alpha[1]/num_planes
    fval += np.sum(tmp_val)*weight
    grad_offset += tmp_grad*weight
    # self normalization
    tmp = np.sum(planes[:,:3]*planes[:,:3], axis=1, keepdims=True) - 1.0
    tmp_val, tmp_grad = loss_func(tmp)
    weight = alpha[2]/num_planes
    fval += np.sum(tmp_val)*weight
    grad_normal += tmp_grad*planes[:,:3]*2*weight
    # parallel list
    pnum = plist.shape[0]
    if pnum>0:
        lst1 = plist[:,0]
        lst2 = plist[:,1]
        tmp = np.sum(planes[lst1,:3]*planes[lst2,:3], axis=1, keepdims=True) - 1.0
        tmp_val, tmp_grad = loss_func(tmp)
        weight = alpha[3]/pnum
        fval += np.sum(tmp_val)*weight
        for i in range(pnum):
            grad_normal[lst1[i],:] += tmp_grad[i]*planes[lst2[i],:3]*weight
            grad_normal[lst2[i],:] += tmp_grad[i]*planes[lst1[i],:3]*weight
    # orthogonal list
    onum = olist.shape[0]
    if onum>0:
        lst1 = olist[:,0]
        lst2 = olist[:,1]
        tmp = np.sum(planes[lst1,:3]*planes[lst2,:3], axis=1, keepdims=True)
        tmp_val, tmp_grad = loss_func(tmp)
        weight = alpha[4]/onum
        fval += np.sum(tmp_val)*weight
        for i in range(onum):
            grad_normal[lst1[i],:] += tmp_grad[i]*planes[lst2[i],:3]*weight
            grad_normal[lst2[i],:] += tmp_grad[i]*planes[lst1[i],:3]*weight
    # contact
    cnum = len(clist)
    if cnum>0:
        for i in range(cnum):
            pid1 = clist[i][0]
            pid2 = clist[i][1]
            plane1 = planes[pid1,:]
            plane2 = planes[pid2,:]
            raydirs = clist[i][2]
            pixnum = raydirs.shape[0]
            if pixnum == 0:
               continue
            tmp = plane2[3]*plane1[:3] - plane1[3]*plane2[:3]
            tmp = tmp.reshape(-1,3)
            tmp1 = np.sum(tmp*raydirs, axis=1, keepdims=True)
            tmp_val, tmp_grad = loss_func(tmp1)
            weight = alpha[5]/cnum/pixnum
            fval += np.sum(tmp_val)*weight
            for j in range(3):
               grad_normal[pid1,j] += np.sum(plane2[3]*raydirs[:,j]*tmp_grad.reshape(-1))*weight
               grad_normal[pid2,j] += np.sum(-plane1[3]*raydirs[:,j]*tmp_grad.reshape(-1))*weight
            grad_offset[pid1] += np.sum(-np.sum(plane2[:3].reshape(-1,3)*raydirs, axis=1, keepdims=True)*tmp_grad)*weight
            grad_offset[pid2] += np.sum(np.sum(plane1[:3].reshape(-1,3)*raydirs, axis=1, keepdims=True)*tmp_grad)*weight
    # put together
    grad = np.concatenate([grad_normal, grad_offset.reshape(-1,1)], axis=1).reshape(-1)
    return fval, grad
if __name__ == '__main__':
    validate_gradients()
