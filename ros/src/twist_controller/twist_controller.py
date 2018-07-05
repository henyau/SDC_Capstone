from pid import PID
from lowpass import LowPassFilter
from yaw_controller import YawController
import rospy

GAS_DENSITY = 2.858
ONE_MPH = 0.44704


class Controller(object):
    def __init__(self, vehicle_mass,
                       fuel_capacity, 
                       brake_deadband, 
                       decel_limit, 
                       accel_limit, 
                       wheel_radius, 
                       wheel_base, 
                       steer_ratio, 
                       max_lat_accel, 
                       max_steer_angle):
        # Controllers
        self.yaw_controller = YawController(wheel_base, steer_ratio,0.1,max_lat_accel,max_steer_angle)
        kp = 0.3
        ki = 0.1
        kd = 0.0
        min_input = 0.0
        max_input = 0.5 
        self.speed_PID_controller = PID(kp, ki , kd, min_input, max_input)

        self.vehicle_mass =  vehicle_mass
        self.fuel_capacity =  fuel_capacity
        self.brake_deadband =  brake_deadband
        self.decel_limit =  decel_limit
        self.accel_limit =  accel_limit
        self.wheel_radius =  wheel_radius
        self.wheel_base = wheel_base
        self.steer_ratio =  steer_ratio
        self.max_lat_accel = max_lat_accel
        self.max_steer_angle =  max_steer_angle

        self.vel_lpf = LowPassFilter(0.5, 0.02)

        self.prev_time = rospy.get_time()

    def control(self, cur_vel, dbw_enabled, linear_vel, angular_vel):
        # Return throttle, brake, steer
        if not dbw_enabled:
            return 0,0,0

        cur_vel = self.vel_lpf.filt(cur_vel)
        #compute steering
        steering = self.yaw_controller.get_steering(linear_vel, angular_vel, cur_vel)
        #compute throttle
        #use a PID controller
        vel_error = linear_vel - cur_vel
        self.prev_vel = cur_vel
        cur_time = rospy.get_time()
        sample_time = cur_time - self.prev_time
        self.prev_time = cur_time

        throttle =  self.speed_PID_controller.step(vel_error, sample_time)
        brake = 0.0
        #car moves forward w/o actutation, apply brakes (Nm)
        if linear_vel == 0.0 and cur_vel<0.1:
            throttle = 0.0
            brake = 700.0
        #should brake    
        elif throttle < 0.1 and vel_error:
            throttle = 0.0
            decel = max(vel_error, self.decel_limit)
            brake = abs(decel)*self.vehicle_mass*self.wheel_radius


        #compute braking
        return throttle, brake, steering
