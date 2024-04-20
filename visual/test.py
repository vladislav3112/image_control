import pybullet as p
import numpy as np
from camera import Camera
import cv2
from scipy.spatial.transform import Rotation as R

IMG_SIDE = 800
IMG_HALF = IMG_SIDE/2
MARKER_LENGTH = 0.1
MARKER_CORNERS_WORLD = np.array(
    [
        [-MARKER_LENGTH/2,MARKER_LENGTH/2,0.0,1],
        [MARKER_LENGTH/2,MARKER_LENGTH/2,0.0,1],
        [MARKER_LENGTH/2,-MARKER_LENGTH/2.0,0.0,1],
        [-MARKER_LENGTH/2,-MARKER_LENGTH/2,0.0,1]
    ]
)

def computeInterMatrix(z, sd0):
    L = np.zeros((8,6))
    for idx in range(4):
        x = sd0[2*idx, 0]
        y = sd0[2*idx+1, 0]
        Z = z[idx]
        L[2*idx] = np.array([-1/Z,0,x/Z,x*y,-(1+x**2),y])
        L[2*idx+1] = np.array([0,-1/Z,y/Z,1+y**2,-x*y,-x])
    return L

def rearrangeCorners(corners):
    idx = 0
    center = [0,0]
    for i in range(4):
        center[0] += corners[i,0]/4
        center[1] += corners[i,1]/4
    for i in range(4):
        p = corners[i]
        if p[0] <= center[0] and p[1] >= center[1]:
            idx = i
            break
    rearranged = np.array([corners[idx], corners[idx-1], corners[idx-2], corners[idx-3]])
    return rearranged

def skew(vec):
    return np.array([[0,-vec[2], vec[1]],
                     [vec[2], 0, -vec[0]],
                     [-vec[1],vec[0], 0]])

def world2img(markerCornersWorld, camera):
    imgCorners = np.zeros((4,2))
    for idx in range(len(markerCornersWorld)):
        mw = markerCornersWorld[idx]
        camMark = np.reshape(camera.viewMatrix,(4,4)).T @ mw
        projMark = np.reshape(camera.projectionMatrix,(4,4)).T @ camMark
        projMark[0] = IMG_HALF*projMark[0]/projMark[3] + IMG_HALF
        projMark[1] = -IMG_HALF*projMark[1]/projMark[3] + IMG_HALF
        imgMark = projMark[:2]
        imgCorners[idx] = imgMark
    return imgCorners

def depth(camera):
    Z = np.zeros(4)
    for idx in range(4):
        mw = MARKER_CORNERS_WORLD[idx]
        camMark = np.reshape(camera.viewMatrix,(4,4)).T @ mw
        Z[idx] = -camMark[2]
    return Z

dt = 1/240
T = 0.5
t = 0

physicsClient = p.connect(p.GUI, options="--background_color_red=1 --background_color_blue=1 --background_color_green=1")#or p.DIRECT for non-graphical version
p.setGravity(0,0,-10)

# add aruco cube and aruco texture
c = p.loadURDF('aruco.urdf', (0.0, 0.0, 0.0), useFixedBase=True)
x = p.loadTexture('aruco_cube.png')
p.changeVisualShape(c, -1, textureUniqueId=x)

#init aruco detector
dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
parameters = cv2.aruco.DetectorParameters()
parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
detector = cv2.aruco.ArucoDetector(dictionary, parameters)

camera = Camera(imgSize = [IMG_SIDE, IMG_SIDE])

euler = [0.0,0.,0]
quat = p.getQuaternionFromEuler(euler)
rotMat = p.getMatrixFromQuaternion(quat)
rotMat = np.reshape(np.array(rotMat),(3,3))
camera.set_new_position([0.0, 0.0, 0.15], rotMat)

wrld = world2img(MARKER_CORNERS_WORLD, camera)

img = camera.get_frame()

corners, markerIds, rejectedCandidates = detector.detectMarkers(img)
sd0 = np.reshape(np.array(corners[0][0]),(8,1))
sd0 = np.array([(s-IMG_HALF)/IMG_HALF for s in sd0])
sd = np.reshape(np.array(corners[0][0]),(8,1)).astype(int)

Z = depth(camera)
Ld0 = computeInterMatrix(Z, sd0)

# f = IMG_HALF/np.tan(np.pi/6)
f = IMG_HALF/np.tan(np.deg2rad(camera.fov)/2)
distCoeffs = np.array([])
cameraMatrix = np.array([[f, 0, IMG_HALF],
                         [0, f, IMG_HALF],
                         [0, 0, 1]])
objPoints = np.array([
    [-MARKER_LENGTH/2, MARKER_LENGTH/2, 0],
    [ MARKER_LENGTH/2, MARKER_LENGTH/2, 0],
    [ MARKER_LENGTH/2,-MARKER_LENGTH/2, 0],
    [-MARKER_LENGTH/2,-MARKER_LENGTH/2, 0]
])
cornersRearranged = rearrangeCorners(corners[0][0])
retval, rvec, tvec = cv2.solvePnP(objPoints, cornersRearranged, cameraMatrix, distCoeffs, flags=cv2.SOLVEPNP_IPPE_SQUARE)
print(tvec)
print(rvec)
r = R.from_rotvec(rvec.flatten())
print(r.as_euler('xyz'))

img = cv2.UMat(img)
cv2.aruco.drawDetectedMarkers(img, corners)
cv2.imshow('init', img)

cv2.waitKey(0)
cv2.destroyAllWindows()

pos = np.array([0.05, 0.05, 0.35])

al = 0.1
bt = 0.0
gm = 0.0

euler = [al,bt,gm]
quat = p.getQuaternionFromEuler(euler)
rotMat = p.getMatrixFromQuaternion(quat)
rotMat = np.reshape(np.array(rotMat),(3,3))
camera.set_new_position(pos, rotMat)

Z = 0.2
img = camera.get_frame()
corners, markerIds, rejectedCandidates = detector.detectMarkers(img)

si = np.reshape(np.array(corners[0][0]),(8,1)).astype(int)
path = [corners[0][0,0].astype(int)]
v = []
while t <= T:
    img = camera.get_frame()

    corners, markerIds, rejectedCandidates = detector.detectMarkers(img)
    img = cv2.UMat(img)
    if (markerIds != None):
    
        cv2.aruco.drawDetectedMarkers(img, corners)
        # for corner debug plot
        prev = path[-1]
        curr = corners[0][0,0].astype(int)
        if (curr[0] != prev[0] or curr[1] != prev[1]):
            path.append(curr)

        s = corners[0][0,0]
        s0 = np.reshape(np.array(corners[0][0]),(8,1))
        s0 = np.array([(ss-IMG_HALF)/IMG_HALF for ss in s0])   

        # Z = pos[2]
        Z = depth(camera)
        L0 = computeInterMatrix(Z, s0)
        
        # L0 = Ld0
        # L0 = (L0+Ld0)/2
        
        L0T = np.linalg.inv(L0.T@L0)@L0.T
        e = s0 - sd0
        coef = 1/10
        w = -coef * L0T @ e
        v = w[:3,0]
        w = w[3:,0]

        corners = corners[0][0]
        # corners = world2img(MARKER_CORNERS_WORLD, camera)
        cornersRearranged = rearrangeCorners(corners)
        retval, rvec, tvec = cv2.solvePnP(objPoints, cornersRearranged, cameraMatrix, distCoeffs, flags=cv2.SOLVEPNP_IPPE_SQUARE)

        if (not np.isnan(tvec[0])):
            tvec_d = [0,0,0.15]
            tvec = tvec.flatten()
            # v = -0.1*((tvec_d-tvec)+skew(tvec)@rvec.flatten())
            # w = 0.1*rvec.flatten()
        

    cv2.imshow('test', img)
    cv2.waitKey(1)

    pos[1] -= v[0]
    pos[0] -= v[1]
    pos[2] -= v[2]
    al -= w[1]
    bt -= w[0]
    gm -= w[2]

    euler = [al,bt,gm]
    quat = p.getQuaternionFromEuler(euler)
    rotMat = p.getMatrixFromQuaternion(quat)
    rotMat = np.reshape(np.array(rotMat),(3,3))
    camera.set_new_position(pos, rotMat)

    p.stepSimulation()
    # t += dt

print('fin: ', tvec, r.as_euler('xyz'))
# draw resulting image
data = camera.get_frame()
img = data[:,:,[2,1,0]]
corners, markerIds, rejectedCandidates = detector.detectMarkers(img)

img = cv2.UMat(img)
# for i in range(4):
#     cv2.circle(img, (sd[2*i,0],sd[2*i+1,0]), 5, (255,0,0),2)
#     cv2.circle(img, (si[2*i,0],si[2*i+1,0]), 5, (0,0,255),2)
#     ii = (i+1)%4
#     cv2.line(img,(sd[2*i,0],sd[2*i+1,0]),(sd[2*ii,0],sd[2*ii+1,0]),(0,0,255),2)
#     cv2.line(img,(si[2*i,0],si[2*i+1,0]),(si[2*ii,0],si[2*ii+1,0]),(255,0,0),2)
# for pt in path:
#     cv2.circle(img, pt, 3, (0,255,0))

cv2.aruco.drawDetectedMarkers(img, corners)
cv2.imshow('fin', img)
cv2.waitKey(0)

p.disconnect()
cv2.destroyAllWindows()