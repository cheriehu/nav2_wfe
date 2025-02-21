#! /usr/bin/env python3
# Copyright 2019 Samsung Research America
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import sys
import time

from action_msgs.msg import GoalStatus
from geometry_msgs.msg import PoseStamped, Pose
# from nav2_msgs.action import FollowWaypoints
from nav2_msgs.action import NavigateToPose
from nav2_msgs.srv import ManageLifecycleNodes
from nav2_msgs.srv import GetCostmap
from nav2_msgs.msg import Costmap
from nav_msgs.msg  import OccupancyGrid
from nav_msgs.msg import Odometry
from std_msgs.msg import Empty, String

from nav2_simple_commander.robot_navigator import BasicNavigator, TaskResult

import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy, QoSHistoryPolicy, QoSReliabilityPolicy
from rclpy.qos import QoSProfile
from rclpy.duration import Duration
from rclpy.parameter import Parameter

from enum import Enum

import numpy as np

import math

from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener

from  tf2_ros import LookupException
import py_trees_ros
import py_trees


OCC_THRESHOLD = 50
MIN_FRONTIER_SIZE = 5

class Costmap2d():
    class CostValues(Enum):
        FreeSpace = 0
        InscribedInflated = 253
        LethalObstacle = 254
        NoInformation = 255
    
    def __init__(self, map):
        self.map = map

    def getCost(self, mx, my):
        return self.map.data[self.__getIndex(mx, my)]

    def getSize(self):
        return (self.map.metadata.size_x, self.map.metadata.size_y)

    def getSizeX(self):
        return self.map.metadata.size_x

    def getSizeY(self):
        return self.map.metadata.size_y

    def __getIndex(self, mx, my):
        return my * self.map.metadata.size_x + mx

class OccupancyGrid2d():
    class CostValues(Enum):
        FreeSpace = 0
        InscribedInflated = 100
        LethalObstacle = 100
        NoInformation = -1

    def __init__(self, map):
        self.map = map

    def getCost(self, mx, my):
        return self.map.data[self.__getIndex(mx, my)]

    def getSize(self):
        return (self.map.info.width, self.map.info.height)

    def getSizeX(self):
        return self.map.info.width

    def getSizeY(self):
        return self.map.info.height

    def mapToWorld(self, mx, my):
        wx = self.map.info.origin.position.x + (mx + 0.5) * self.map.info.resolution
        wy = self.map.info.origin.position.y + (my + 0.5) * self.map.info.resolution

        return (wx, wy)

    def worldToMap(self, wx, wy):
        if (wx < self.map.info.origin.position.x or wy < self.map.info.origin.position.y):
            raise Exception("World coordinates out of bounds")

        mx = int((wx - self.map.info.origin.position.x) / self.map.info.resolution)
        my = int((wy - self.map.info.origin.position.y) / self.map.info.resolution)
        
        if  (my > self.map.info.height or mx > self.map.info.width):
            raise Exception("Out of bounds")

        return (mx, my)

    def __getIndex(self, mx, my):
        return my * self.map.info.width + mx

class FrontierCache():
    cache = {}

    def getPoint(self, x, y):
        idx = self.__cantorHash(x, y)

        if idx in self.cache:
            return self.cache[idx]

        self.cache[idx] = FrontierPoint(x, y)
        return self.cache[idx]

    def __cantorHash(self, x, y):
        return (((x + y) * (x + y + 1)) / 2) + y

    def clear(self):
        self.cache = {}

class FrontierPoint():
    def __init__(self, x, y):
        self.classification = 0
        self.mapX = x
        self.mapY = y

def centroid(arr):
    arr = np.array(arr)
    length = arr.shape[0]
    sum_x = np.sum(arr[:, 0])
    sum_y = np.sum(arr[:, 1])
    return sum_x/length, sum_y/length

def findFree(mx, my, costmap):
    fCache = FrontierCache()

    bfs = [fCache.getPoint(mx, my)]

    while len(bfs) > 0:
        loc = bfs.pop(0)

        if costmap.getCost(loc.mapX, loc.mapY) == OccupancyGrid2d.CostValues.FreeSpace.value:
            return (loc.mapX, loc.mapY)

        for n in getNeighbors(loc, costmap, fCache):
            if n.classification & PointClassification.MapClosed.value == 0:
                n.classification = n.classification | PointClassification.MapClosed.value
                bfs.append(n)

    return (mx, my)

def getFrontier(current_pose:Pose, map:OccupancyGrid2d, costmap:OccupancyGrid2d, logger):
    fCache = FrontierCache()

    fCache.clear()

    mx, my = map.worldToMap(current_pose.position.x, current_pose.position.y)

    freePoint = findFree(mx, my, map)
    start = fCache.getPoint(freePoint[0], freePoint[1])
    start.classification = PointClassification.MapOpen.value
    mapPointQueue = [start]

    frontiers = []

    while len(mapPointQueue) > 0:
        p = mapPointQueue.pop(0)

        if p.classification & PointClassification.MapClosed.value != 0:
            continue

        if isFrontierPoint(p, map, costmap, fCache):
            p.classification = p.classification | PointClassification.FrontierOpen.value
            frontierQueue = [p]
            newFrontier = []

            while len(frontierQueue) > 0:
                q = frontierQueue.pop(0)

                if q.classification & (PointClassification.MapClosed.value | PointClassification.FrontierClosed.value) != 0:
                    continue

                if isFrontierPoint(q, map, costmap, fCache):
                    newFrontier.append(q)

                    for w in getNeighbors(q, map, fCache):
                        if w.classification & (PointClassification.FrontierOpen.value | PointClassification.FrontierClosed.value | PointClassification.MapClosed.value) == 0:
                            w.classification = w.classification | PointClassification.FrontierOpen.value
                            frontierQueue.append(w)

                q.classification = q.classification | PointClassification.FrontierClosed.value

            
            newFrontierCords = []
            for x in newFrontier:
                x.classification = x.classification | PointClassification.MapClosed.value
                newFrontierCords.append(map.mapToWorld(x.mapX, x.mapY))

            if len(newFrontier) > MIN_FRONTIER_SIZE:
                frontiers.append(centroid(newFrontierCords))

        for v in getNeighbors(p, map, fCache):
            if v.classification & (PointClassification.MapOpen.value | PointClassification.MapClosed.value) == 0:
                if any(map.getCost(x.mapX, x.mapY) == OccupancyGrid2d.CostValues.FreeSpace.value for x in getNeighbors(v, map, fCache)):
                    v.classification = v.classification | PointClassification.MapOpen.value
                    mapPointQueue.append(v)

        p.classification = p.classification | PointClassification.MapClosed.value
    return frontiers
        

def getNeighbors(point, map, fCache):
    neighbors = []

    for x in range(point.mapX - 1, point.mapX + 2):
        for y in range(point.mapY - 1, point.mapY + 2):
            if (x > 0 and x < map.getSizeX() and y > 0 and y < map.getSizeY()):
                neighbors.append(fCache.getPoint(x, y))

    return neighbors

def isFrontierPoint(point, map, costmap, fCache):
    if map.getCost(point.mapX, point.mapY) != OccupancyGrid2d.CostValues.NoInformation.value:
        return False
    

    hasFree = False
    # Checks two things
    # 1. whether neighbors are in high cost areas
    # 2. whether neighbors have are known freespace
    for n in getNeighbors(point, map, fCache):
        cost = costmap.getCost(n.mapX, n.mapY)
        mapCost = map.getCost(n.mapX, n.mapY)

        # If neighbors have high cost in global costmap, don't count as frontier point
        if cost > OCC_THRESHOLD:
            return False

        # If neighbors are a known freespace in the map, count as frontier point
        if mapCost == OccupancyGrid2d.CostValues.FreeSpace.value:
            hasFree = True

    return hasFree

class PointClassification(Enum):
    MapOpen = 1
    MapClosed = 2
    FrontierOpen = 4
    FrontierClosed = 8

class WaypointFollowerTest(Node):

    def __init__(self):
        super().__init__(node_name='nav2_waypoint_tester', namespace='')
        self.waypoints = None
        self.readyToMove = True
        self.currentPose = None
        self.lastWaypoint = None
        # self.action_client = ActionClient(self, FollowWaypoints, 'FollowWaypoints')
        self.action_client = ActionClient(self, NavigateToPose, 'navigate_to_pose')
        self.initial_pose_pub = self.create_publisher(PoseStamped,
                                                      'initialpose', 10)
        
        # self.costmapClient = self.create_client(GetCostmap, '/global_costmap/get_costmap')
        # while not self.costmapClient.wait_for_service(timeout_sec=1.0):
        #     self.info_msg('service not available, waiting again...')
        self.initial_pose_received = False
        self.goal_handle = None

        # pose_qos = QoSProfile(
        #   durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
        #   reliability=QoSReliabilityPolicy.RELIABLE,
        #   # reliability=QoSReliabilityPolicy.BEST_EFFORT,
        #   history=QoSHistoryPolicy.KEEP_LAST,
        #   depth=1)

        controller_qos_profile = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
            depth=10
        )

        # self.done_scanning_sub = self.create_subscription(
        #     Empty,
        #     "/done_scanning",
        #     self.doneScanningCallback,
        #     10
        # )
        
        self.controller_publisher_ = self.create_publisher(String, "/controller_selector", controller_qos_profile)

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        tf_future = self.tf_buffer.wait_for_transform_async('map', 'base_link', rclpy.time.Time())
        rclpy.spin_until_future_complete(self, tf_future)
        self.init_pose = self.getCurrentPose()

        self.get_current_pose_timer = self.create_timer(1.0, self.poseCallback)
        # self.model_pose_sub = self.create_subscription(Odometry,
        #   '/odom', self.poseCallback, 10) # pose_qos)

        # self.map_received = False

        self.costmapSub = self.create_subscription(OccupancyGrid, '/global_costmap/costmap', self.costmapCallback, 10)
        self.mapSub = self.create_subscription(OccupancyGrid, '/map', self.occupancyGridCallback, 10)
        self.costmap = None
        self.map = None

        self.get_logger().info('Running Waypoint Test')

    def occupancyGridCallback(self, msg):
        if self.costmap == None:
            return
        # self.get_logger().info("occupancyGridCallback")
        self.map = OccupancyGrid2d(msg)
        # self.map_received = True
        self.moveToFrontiers()
    
    # def doneScanningCallback(self, msg):
    #     if not self.map_received:
    #         return
        # self.moveToFrontiers()
    
    def moveToFrontiers(self):
        self.get_logger().info("moveToFrontiers")
        frontiers = getFrontier(self.currentPose, 
                                map=self.map, 
                                costmap=self.costmap, 
                                logger=self.get_logger())

        if len(frontiers) == 0:
            self.info_msg('No More Frontiers')
            return

        location = None
        largestDist = 0
        smallestDist = 1e9
        for f in frontiers:
            dist = math.sqrt(((f[0] - self.currentPose.position.x)**2) + ((f[1] - self.currentPose.position.y)**2))
            if dist > largestDist and dist > 0.5:
                largestDist = dist
                location = [f] 
        
        #worldFrontiers = [self.costmap.mapToWorld(f[0], f[1]) for f in frontiers]
        self.info_msg(f'World points {location}')
        self.setWaypoints(location)

        # action_request = FollowWaypoints.Goal()
        action_request = NavigateToPose.Goal()
        # action_request.poses = self.waypoints
        action_request.pose = self.waypoints[0]

        self.info_msg('Sending goal request...')
        send_goal_future = self.action_client.send_goal_async(action_request, self._feedbackCallback)
        
        # Add done callback instead of spin_until_future_complete
        send_goal_future.add_done_callback(self.goal_response_callback)

    def goal_response_callback(self, future):
        try:
            goal_handle = future.result()
            if not goal_handle.accepted:
                self.error_msg('Goal was rejected by server!')
                return

            self.info_msg('Goal accepted!')
            # Save the handle so we can cancel if needed, etc.
            self.goal_handle = goal_handle

            # Get the result asynchronously
            self.get_result_future = self.goal_handle.get_result_async()
            self.get_result_future.add_done_callback(self.get_result_callback)
        except Exception as e:
            self.error_msg(f'Service call failed: {repr(e)}')

    def get_result_callback(self, future):
        """Called when the action server has a result (success, failure, or canceled)."""
        try:
            result = future.result().result
            status = future.result().status
            self.info_msg(f"Goal finished with status [{status}]")

            # If you want to re-invoke frontier search after finishing:
            # self.moveToFrontiers()

        except Exception as e:
            self.error_msg(f'get_result_callback exception: {repr(e)}')

    def _feedbackCallback(self, msg):
        self.feedback = msg.feedback
        print(self.feedback.distance_remaining)
        controller_msg = String()
        if msg.feedback.distance_remaining < 0.5:
            controller_msg.data = "FollowPathOmni"
        self.controller_publisher_.publish(controller_msg)

    def costmapCallback(self, msg):
        # self.get_logger().info("costmapCallback")
        self.costmap = OccupancyGrid2d(msg)

        # unknowns = 0
        # for x in range(0, self.costmap.getSizeX()):
        #     for y in range(0, self.costmap.getSizeY()):
        #         if self.costmap.getCost(x, y) == 255:
        #             unknowns = unknowns + 1
        # self.get_logger().info(f'Unknowns {unknowns}')
        # self.get_logger().info(f'Got Costmap {len(getFrontier(None, self.costmap, self.get_logger()))}')

    # def dumpCostmap(self):
    #     costmapReq = GetCostmap.Request()
    #     self.get_logger().info('Requesting Costmap')
    #     costmap = self.costmapClient.call(costmapReq)
    #     self.get_logger().info(f'costmap resolution {costmap.specs.resolution}')

    def setInitialPose(self):
        # self.init_pose = PoseWithCovarianceStamped()
        # self.init_pose.pose.pose.position.x = pose[0]
        # self.init_pose.pose.pose.position.y = pose[1]
        # self.init_pose.header.frame_id = 'map'
        self.currentPose = self.init_pose
        # self.publishInitialPose()
        self.initial_pose_received = True
        time.sleep(5)

    def poseCallback(self):
        self.currentPose = self.getCurrentPose()
        # if (not self.initial_pose_received):
        #   self.info_msg('Received amcl_pose')
        # self.currentPose = msg.pose.pose
        # self.initial_pose_received = True

    def getCurrentPose(self):
        self.get_logger().debug("getCurrentPose")
        transform = self.tf_buffer.lookup_transform(
            'map',
            'base_link',
            rclpy.time.Time(), 
            Duration(seconds=0.5)
        )
        self.pose = PoseStamped()
        self.pose.pose.position.x = transform.transform.translation.x
        self.pose.pose.position.y = transform.transform.translation.y
        self.pose.pose.position.z = transform.transform.translation.z
        self.pose.pose.orientation = transform.transform.rotation
        self.pose.header = transform.header
        return self.pose.pose

    def setWaypoints(self, waypoints):
        self.waypoints = []
        for wp in waypoints:
            msg = PoseStamped()
            msg.header.frame_id = 'map'
            msg.pose.position.x = wp[0]
            msg.pose.position.y = wp[1]
            msg.pose.orientation.w = 1.0
            self.waypoints.append(msg)

    def publishInitialPose(self):
        self.initial_pose_pub.publish(self.pose)

    def cancel_goal(self):
        cancel_future = self.goal_handle.cancel_goal_async()
        rclpy.spin_until_future_complete(self, cancel_future)

    def info_msg(self, msg: str):
        self.get_logger().info(msg)

    def warn_msg(self, msg: str):
        self.get_logger().warn(msg)

    def error_msg(self, msg: str):
        self.get_logger().error(msg)

class ExplorerBehaviour(py_trees.behaviour.Behaviour):
    """
    Write goal pose to blackboard
    Behaviour-ifying the WaypointFollwerTest node.
    Read camera_map from blackboard 
    """
    def __init__(
            self,
            name: str="ExplorerBehaviour",
            map_topic: str="/combined_map"
    ):
        super(ExplorerBehaviour, self).__init__(name=name)
        # self.waypoint_follower = WaypointFollowerTest()
        self.map_topic = map_topic
        self._initialization_ok = False
        self.blackboard = self.attach_blackboard_client(name="ExplorerBehaviour")
        self.blackboard.register_key("goal_pose", py_trees.common.Access.WRITE)
        self.blackboard.register_key("camera_map", py_trees.common.Access.READ)
        self.blackboard.register_key("status", py_trees.common.Access.READ)
        self.blackboard.register_key("status", py_trees.common.Access.WRITE)
        self.blackboard.register_key("internal_semantic_map", py_trees.common.Access.READ)
        self.blackboard.register_key("semantic_map", py_trees.common.Access.WRITE)

    def setup(self, **kwargs):
        try:
            self.node = kwargs['node']
            # self.node.__init__(node_name='ExplorerBehaviour',
            #                    parameter_overrides=[
            #                         Parameter('use_sim_time', 
            #                         Parameter.Type.BOOL, 
            #                         True)
            #                         ])

        except KeyError as e:
            error_message = "didn't find 'node' in setup's kwargs [{}][{}]".format(self.qualified_name)
            raise KeyError(error_message) from e  # 'direct cause' traceability
        
        self.waypoints = None

        self.navigator = BasicNavigator()

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self.node)
        tf_future = self.tf_buffer.wait_for_transform_async('map', 'base_link', rclpy.time.Time())
        rclpy.spin_until_future_complete(self.node, tf_future)
        self.init_pose = self.getCurrentPose()
        self.currentPose = self.getCurrentPose()

        self.costmapSub = self.node.create_subscription(OccupancyGrid, '/global_costmap/costmap', self.costmapCallback, 10)
        # self.mapSub = self.node.create_subscription(OccupancyGrid, self.map_topic, self.mapCallback, 10)
        self.costmap = None
        self.map = None

    def getCurrentPose(self):
        self.node.get_logger().debug("getCurrentPose")
        try:
            transform = self.tf_buffer.lookup_transform(
                'map',
            'base_link',
                rclpy.time.Time(), 
            Duration(seconds=0.5)
            )
            self.pose = PoseStamped()
            self.pose.pose.position.x = transform.transform.translation.x
            self.pose.pose.position.y = transform.transform.translation.y
            self.pose.pose.position.z = transform.transform.translation.z
            self.pose.pose.orientation = transform.transform.rotation
            self.pose.header = transform.header
            return self.pose.pose
        except LookupException:
            print("LOOKUP EXCEPTION")
            return None

    def costmapCallback(self, msg):
        # print('costmapCallback')
        self.costmap = OccupancyGrid2d(msg)
        
    def mapCallback(self, msg):
        print('mapCallback')
        self.map = OccupancyGrid2d(msg)

        pose = self.getNextFrontier()
        if pose == None:
            print("pose = None")
            return
        goal_request = NavigateToPose.Goal()
        goal_request.pose = pose
        return goal_request

    def setWaypoints(self, waypoints):
        self.waypoints = []
        for wp in waypoints:
            msg = PoseStamped()
            msg.header.frame_id = 'map'
            msg.pose.position.x = wp[0]
            msg.pose.position.y = wp[1]
            msg.pose.orientation.w = 1.0
            path = self.navigator.getPath(start=msg, goal=msg, use_start=False)
            # print(path)
            if path == None:
                print("PATH IS NONE")
                continue
            self.waypoints.append(msg)

    def getNextFrontier(self):
        self.getCurrentPose()
        if self.map == None or self.costmap == None:
            return

        frontiers = getFrontier(self.currentPose, 
                                map=self.map, 
                                costmap=self.costmap, 
                                logger=self.node.get_logger())

        if len(frontiers) == 0:
            self.node.get_logger().info('No More Frontiers')
            return

        location = []
        largestDist = 0
        smallestDist = 1e9
        for f in frontiers:
            
            msg = PoseStamped()
            msg.header.frame_id = 'map'
            msg.pose.position.x = f[0]
            msg.pose.position.y = f[1]
            msg.pose.orientation.w = 1.0
            path = self.navigator.getPath(start=msg, goal=msg, use_start=False)

            if path == None:
                # input()
                print("PATH IS NONE")
                continue
            
            dist = math.sqrt(((f[0] - self.currentPose.position.x)**2) + ((f[1] - self.currentPose.position.y)**2))

            if dist > largestDist and dist > 0.2:
                largestDist = dist
                location.append(f) 

        #worldFrontiers = [self.costmap.mapToWorld(f[0], f[1]) for f in frontiers]
        self.node.get_logger().info(f'World points {location}')
        self.setWaypoints(location)

        if len(self.waypoints) == 0:
            self.node.get_logger().info('No More Frontiers')
            return

        return self.waypoints[0]

    def initialise(self):
        if not self.blackboard.exists("camera_map"):
            return
        
        # if self.blackboard.status != "EXPLORE":
        #     return

        # print(self.map)
        # print(self.costmap)
        
        self._initialization_ok = True

    def update(self):
        # TODO: add in max retries?
        if not self._initialization_ok:
            print("explorer not self._initialization_ok")
            return py_trees.common.Status.FAILURE
        
        # if self.blackboard.status != "EXPLORE":
        #     return py_trees.common.Status.FAILURE

        else:
            print("explorer running")
            goal_request = self.mapCallback(self.blackboard.camera_map)
            if goal_request == None:
                # self.blackboard.status = "EXPLORATION_DONE"
                self.blackboard.semantic_map = self.blackboard.internal_semantic_map
            else:
                self.blackboard.goal_pose = goal_request
                # self.status = "MOVING"
            return py_trees.common.Status.SUCCESS

        ## comment out below
        self.blackboard.semantic_map = self.blackboard.internal_semantic_map
        return py_trees.common.Status.SUCCESS

    
def main(argv=sys.argv[1:]):
    rclpy.init()

    # wait a few seconds to make sure entire stacks are up
    #time.sleep(10)

    # wps = [[-0.52, -0.54], [0.58, -0.55], [0.58, 0.52]]
    # starting_pose = [-2.0, -0.5]

    test = WaypointFollowerTest()
    #test.dumpCostmap()
    # test.setWaypoints(wps)

    retry_count = 0
    retries = 2000
    while not test.initial_pose_received and retry_count <= retries:
        retry_count += 1
        test.info_msg('Setting initial pose')
        test.setInitialPose()
        test.info_msg('Waiting for amcl_pose to be received')
        rclpy.spin_once(test, timeout_sec=1.0)  # wait for poseCallback

    while test.map == None:
        test.info_msg('Getting initial map')
        rclpy.spin_once(test, timeout_sec=1.0)

    # test.moveToFrontiers()

    rclpy.spin(test)

if __name__ == '__main__':
    main()
