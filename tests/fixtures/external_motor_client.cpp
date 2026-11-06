#include <base_control/motor_iface.hpp>
#include <chrono>
#include <iostream>
#include <json.hpp>
#include <thread>

int main() {
  try {
    // These are the two motors in the test rig, on separate buses.
    MotorManager wheels(can::CAN0, {MotorController::ODRIVE,
        {{12, {.pos = MotorPosition::DRIVE_FRONT_LEFT, .max_torque = 2}}}});
    MotorManager arm(can::CAN1, {MotorController::ODRIVE,
        {{21, {.pos = MotorPosition::ARM_YAW, .max_torque = 2}}}});
    if (!wheels.ping(robot_time::seconds(1)) || !arm.ping(robot_time::seconds(1))) return 2;
    ControllerParams velocity{.mode=ControlMode::VELOCITY, .kp=5, .max_torque=1};
    ControllerParams position{.mode=ControlMode::POSITION, .kp=5, .kd=1, .max_torque=1};
    wheels.configure(velocity);
    arm.configure(position);
    wheels.enable();
    arm.enable();
    wheels.id(12).reference.velocity_ref = 2;
    arm.id(21).reference.position_ref = .4;
    for (int step=0; step<150; ++step) {
      wheels.emit_commands();
      arm.emit_commands();
      std::this_thread::sleep_for(std::chrono::milliseconds(2));
    }
    wheels.get_feedback();
    arm.get_feedback();
    std::cout << nlohmann::json({{"wheel", wheels.id(12).feedback}, {"arm", arm.id(21).feedback}}).dump() << std::endl;
  } catch (const std::exception &error) {
    std::cerr << error.what() << std::endl;
    return 1;
  }
}
