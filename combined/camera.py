import pybullet as pb
import numpy as np

class Camera:
    def __init__(self, cameraEyePosition=np.array([0.5, 0.5, 3.0]), imgSize=[250,250]):
        self.size = {
            'width': imgSize[0],
            'height': imgSize[1]
        }
        self.targetVec = np.array([0,0,-1])
        self.upVec = np.array([1,0,0])
        self.viewMatrix = pb.computeViewMatrix(
            cameraEyePosition=cameraEyePosition,
            cameraTargetPosition = cameraEyePosition + self.targetVec,
            cameraUpVector=self.upVec
        )
        self.fov = 45
        self.projectionMatrix = pb.computeProjectionMatrixFOV(
            fov=self.fov,
            aspect=1.0,
            nearVal=0.1,
            farVal=100
        )
        self.cam_image_kwargs = {
            **self.size,
            'viewMatrix': self.viewMatrix,
            'projectionMatrix': self.projectionMatrix,
            'flags': pb.ER_NO_SEGMENTATION_MASK,
            'renderer': pb.ER_TINY_RENDERER
        }

    def set_new_height(self, h):
        self.__init__(size=self.size, height=h)

    def set_new_position(self, pos, rotMat = np.eye(3)):
        self.viewMatrix = pb.computeViewMatrix(
            cameraEyePosition = pos,
            cameraTargetPosition = pos + (rotMat @ self.targetVec).flatten(),
            cameraUpVector = (rotMat @ self.upVec).flatten()
        )
        self.cam_image_kwargs = {
            **self.size,
            'viewMatrix': self.viewMatrix,
            'projectionMatrix': self.projectionMatrix,
            'flags': pb.ER_NO_SEGMENTATION_MASK,
            'renderer': pb.ER_TINY_RENDERER
        }

    def get_frame(self):
        pbImg = pb.getCameraImage(**self.cam_image_kwargs)[2]
        cvImg = pbImg[:,:,[2,1,0]]
        return cvImg
