from math import sqrt
import numpy as np
from scipy import linalg


def angle_calculation(ap_axis, coil_axis):
    """
    Calculate angle between two given axis (in degrees)

    :param ap_axis: anterior posterior axis represented
    :param coil_axis: tms coil axis
    :return: angle between the two given axes
    """

    ap_axis = np.array([ap_axis[0], ap_axis[1]])
    coil_axis = np.array([float(coil_axis[0]), float(coil_axis[1])])
    angle = np.rad2deg(np.arccos((np.dot(ap_axis, coil_axis))/(
        np.linalg.norm(ap_axis)*np.linalg.norm(coil_axis))))

    return float(angle)


def base_creation(fiducials):
    """

    """

    p1 = fiducials[0, :]
    p2 = fiducials[1, :]
    p3 = fiducials[2, :]

    sub1 = p2 - p1
    sub2 = p3 - p1
    lamb = (sub1[0]*sub2[0]+sub1[1]*sub2[1]+sub1[2]*sub2[2])/np.dot(sub1, sub1)

    q = p1 + lamb*sub1
    g1 = p1 - q
    g2 = p3 - q

    if not g1.any():
        g1 = p2 - q

    g3 = np.cross(g2, g1)

    g1 = g1/sqrt(np.dot(g1, g1))
    g2 = g2/sqrt(np.dot(g2, g2))
    g3 = g3/sqrt(np.dot(g3, g3))

    m = np.matrix([[g1[0], g1[1], g1[2]],
                   [g2[0], g2[1], g2[2]],
                   [g3[0], g3[1], g3[2]]])

    q.shape = (3, 1)
    q = np.matrix(q.copy())
    m_inv = m.I

    # print"M: ", m
    # print"q: ", q

    return m, q, m_inv

def calculate_fre(fiducials,minv,n,q1,q2):

    """
    Calculate the Fiducial Registration Error for neuronavigation.

    :param fiducials: array of 6 rows (image and tracker fiducials) and 3 columns (x, y, z) with coordinates
    :param minv: inverse matrix given by base creation
    :param n: base change matrix given by base creation
    :param q1: origin of first base
    :param q2: origin of second base
    :return: float number of fiducial registration error
    """

    print fiducials

    img = [fiducials[0, :],fiducials[1, :],fiducials[2, :]]
    trk = [fiducials[3, :],fiducials[4, :],fiducials[5, :]]
    result = []
    dist = np.zeros([3,1])

    ponto1 = np.array(trk[0])
    ponto1.shape = (3,1)
    ponto1 = np.matrix(ponto1.copy())
    ponto2 = np.matrix(trk[1])
    ponto2.shape = (3,1)
    ponto2 = np.matrix(ponto2.copy())
    ponto3 = np.matrix(trk[2])
    ponto3.shape = (3,1)
    ponto3 = np.matrix(ponto3.copy())

    imagem1 = np.array(q1 + (minv * n) * (ponto1 - q2))
    imagem2 = np.array(q1 + (minv * n) * (ponto2 - q2))
    imagem3 = np.array(q1 + (minv * n) * (ponto3 - q2))

    ED1=np.sqrt((((imagem1[0]-img[0][0])**2) + ((imagem1[1]-img[0][1])**2) +((imagem1[2]-img[0][2])**2)))
    ED2=np.sqrt((((imagem2[0]-img[1][0])**2) + ((imagem2[1]-img[1][1])**2) +((imagem2[2]-img[1][2])**2)))
    ED3=np.sqrt((((imagem3[0]-img[2][0])**2) + ((imagem3[1]-img[2][1])**2) +((imagem3[2]-img[2][2])**2)))

    FRE = float(np.sqrt((ED1**2 + ED2**2 + ED3**2)/3))

    return FRE



def flip_x(point):
    """
    Flip coordinates of a vector according to X axis
    Coronal Images do not require this transformation - 1 tested
    and for this case, at navigation, the z axis is inverted

    It's necessary to multiply the z coordinate by (-1). Possibly
    because the origin of coordinate system of imagedata is
    located in superior left corner and the origin of VTK scene coordinate
    system (polygonal surface) is in the interior left corner. Second
    possibility is the order of slice stacking

    :param point: list of coordinates x, y and z
    :return: flipped coordinates
    """

    # TODO: check if the Flip function is related to the X or Y axis

    point = np.matrix(point + (0,))
    point[0, 2] = -point[0, 2]

    m_rot = np.matrix([[1.0, 0.0, 0.0, 0.0],
                      [0.0, -1.0, 0.0, 0.0],
                      [0.0, 0.0, -1.0, 0.0],
                      [0.0, 0.0, 0.0, 1.0]])
    m_trans = np.matrix([[1.0, 0, 0, -point[0, 0]],
                        [0.0, 1.0, 0, -point[0, 1]],
                        [0.0, 0.0, 1.0, -point[0, 2]],
                        [0.0, 0.0, 0.0, 1.0]])
    m_trans_return = np.matrix([[1.0, 0, 0, point[0, 0]],
                               [0.0, 1.0, 0, point[0, 1]],
                               [0.0, 0.0, 1.0, point[0, 2]],
                               [0.0, 0.0, 0.0, 1.0]])
        
    point_rot = point*m_trans*m_rot*m_trans_return
    x, y, z = point_rot.tolist()[0][:3]

    return x, y, z

def create_matrix(fiducials):

    img = [fiducials[0, :],fiducials[1, :],fiducials[2, :],fiducials[3, :]]
    trk = [fiducials[4, :],fiducials[5, :],fiducials[6, :],fiducials[7, :]]

    from math import factorial
    num_points = len(img)
    C = factorial(num_points)/(factorial(3)*factorial(num_points - 3))
    l=0
    img_pts=[]
    trk_pts=[]
    e=[]
    for i in range(0,num_points-2):
        for j in range(i+1,num_points-1):
            for k in range(j+1,num_points):

                img_pts.insert(l,[img[i],img[j],img[k]])
                trk_pts.insert(l,[trk[i],trk[j],trk[k]])

                m, q1, minv = base_creation(np.array([img[i],img[j],img[k]]))
                n, q2, ninv = base_creation(np.array([trk[i],trk[j],trk[k]]))

                FRE = calculate_fre(np.array([img[i],img[j],img[k],trk[i],trk[j],trk[k]]),minv,n,q1,q2)
                e.insert(l,[FRE])
                l=l+1
                print FRE

    min = np.argmin(e)
    print np.array(img_pts[min])
    print np.array(trk_pts[min])
    m,q1,minv = base_creation(np.array(img_pts[min]))
    n,q2,ninv = base_creation(np.array(trk_pts[min]))
    fre = np.array(e[min])

    return m,q1,minv,n,q2,ninv, float(fre)
