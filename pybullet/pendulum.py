import pybullet as p
import time
import pybullet_data
import numpy as np

# TASKS FROM 01.03.2024:
# 1  set maxTime
# 2  plots (pos, vel)
# 3  position control based on p.VELOCITY_CONTROL (proportional regulator)
# 4  position control based on p.TORQUE_CONTROL (PI-regulator)
# 5* compare plots of pybullet and our own odeint and figure out the source of errors and fix it
# 6* figure out how to add control to our own integration script

dt = 1/240 # pybullet simulation step
q0 = 0.5  # starting position (radian)
qd = 0.5

xd = 0.1
zd = 1.0

L = 0.5
Z0 = 1.5
pos = q0
maxTime = 50
logTime = np.arange(0.0, maxTime, dt)
sz = logTime.size
logPos = np.zeros(sz)
logPos[0] = q0
logVel = np.zeros(sz)
# logVel[0] = 0

jointIndices = [1,3]
eefLinkIdx = 4
eefLinkIdx = 2

physicsClient = p.connect(p.GUI) # or p.DIRECT for non-graphical version
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0,0,-10)
planeId = p.loadURDF("plane.urdf")
boxId = p.loadURDF("pybullet/simple.urdf", useFixedBase=True)

# get rid of all the default damping forces
# p.changeDynamics(boxId, 1, linearDamping=0, angularDamping=0)
# p.changeDynamics(boxId, 2, linearDamping=0, angularDamping=0)
# p.changeDynamics(boxId, 3, linearDamping=0, angularDamping=0)

numJoints = p.getNumJoints(boxId)
for idx in range(numJoints):
    print(f"{idx} {p.getJointInfo(boxId, idx)[1]} {p.getJointInfo(boxId, idx)[12]}")

# go to the starting position
# p.setJointMotorControl2(bodyIndex=boxId, jointIndex=1, targetPosition=q0, controlMode=p.POSITION_CONTROL)
p.setJointMotorControlArray(bodyIndex=boxId, jointIndices=jointIndices, targetPositions=[q0,q0], controlMode=p.POSITION_CONTROL)
for _ in range(1000):
    p.stepSimulation()

# turn off the motor for the free motion
p.setJointMotorControlArray(bodyIndex=boxId, jointIndices=jointIndices, targetVelocities=[0,0], controlMode=p.VELOCITY_CONTROL, forces=[0,0])

# velocity and torque controllers
# p.setJointMotorControl2(bodyIndex=boxId, jointIndex=1, targetVelocity=0.0, controlMode=p.VELOCITY_CONTROL)
# p.setJointMotorControl2(bodyIndex=boxId, jointIndex=1, force=0.0, controlMode=p.TORQUE_CONTROL)
idx = 1
kp = -100
ki = -50
kd = -10
e_int = 0
e_prev = 0
for t in logTime[1:]:
    e = pos-qd
    e_int += e*dt
    u = kp*e + ki*e_int + kd*(e - e_prev)/dt
    e_prev = e
    # p.setJointMotorControl2(bodyIndex=boxId, jointIndex=1, targetVelocity=u, controlMode=p.VELOCITY_CONTROL)
    #p.setJointMotorControl2(bodyIndex=boxId, jointIndex=1, force=u, controlMode=p.TORQUE_CONTROL)
    p.stepSimulation()
    jointState = p.getJointState(boxId, 1)
    th1 = jointState[0]
    dth1 = jointState[1]
    logPos[idx] = th1
    logVel[idx-1] = dth1
    idx += 1

    xTheor1 = -np.sin(pos)*L
    zTheor1 = Z0 - np.cos(pos)*L

    th2 = p.getJointState(boxId, 3)[0]
    dth2 = p.getJointState(boxId, 3)[1]
    # xTheor2 = xTheor1 - L*np.sin(pos+pos2)
    # zTheor2 = zTheor1 - L*np.cos(pos+pos2)
    
    # linkState = p.getLinkState(boxId, linkIndex=2)
    # xSim1 = linkState[0][0]
    # zSim1 = linkState[0][2]
    # print(f"THEOR1: {xTheor1} {zTheor1}")
    # print(f"SIMUL1: {xSim1} {zSim1}")

    linkState = p.getLinkState(boxId, linkIndex=4)
    xSim2 = linkState[0][0]
    zSim2 = linkState[0][2]
    # print(f"THEOR2: {xTheor2} {zTheor2}")
    print(f"SIMUL2: {xSim2} {zSim2}")

    J = np.array([[-L*np.cos(th1)-L*np.cos(th1+th2), -L*np.cos(th1+th2)], 
                  [L*np.sin(th1)+L*np.sin(th1+th2), L*np.sin(th1+th2)]])
    dth = np.array([[dth1],[dth2]])
    
    w = 10*np.linalg.inv(J) @ -np.array([[xSim2-xd],[zSim2-zd]])
    # print(w)
    p.setJointMotorControlArray(bodyIndex=boxId, jointIndices=jointIndices, targetVelocities=[w[0,0],w[1,0]], controlMode=p.VELOCITY_CONTROL)

    time.sleep(dt)
    
p.disconnect()

import matplotlib.pyplot as plt

plt.subplot(2,1,1)
plt.grid(True)
plt.plot(logTime, logPos, label = "simPos")
plt.plot([logTime[0],logTime[-1]], [qd,qd], '--r', label = "refPos")
plt.legend()

plt.subplot(2,1,2)
plt.grid(True)
plt.plot(logTime, logVel, label = "simVel")
plt.plot([logTime[0],logTime[-1]], [0,0], '--r', label = "refVel")
plt.legend()

plt.show()