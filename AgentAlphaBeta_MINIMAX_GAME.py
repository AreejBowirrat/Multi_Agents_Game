from Agent import Agent, AgentGreedy
from WarehouseEnv import WarehouseEnv, manhattan_distance
import random
from time import time

INF = 99
MINIMUM_TIME = 0.1
EPSILON_LIMIT = 0.1


def is_charging_station(env: WarehouseEnv, state):
    for station in env.charge_stations:
        stat_pos = station.position[1] * 5 + station.position[0]

        if stat_pos == state:
            return True
    return False

def game_is_over(env, robot_id):
    robot = env.get_robot(robot_id)
    other_robot = env.get_robot((robot_id + 1) % 2)
    if robot.battery == 0 and other_robot.battery == 0 and other_robot.credit > robot.credit:
        return -INF, True

    elif robot.battery == 0 and other_robot.battery == 0 and robot.credit > other_robot.credit:
        return INF, True

    return 0, False

# checks if there is enough battery in the robot
def battery_sufficient(robot):
    steps_to_package = manhattan_distance(robot.position, robot.package.position)
    if steps_to_package < robot.battery:
        return True
    return False


def decide_best_robot(env, robot, other_pack_0_dist, other_pack_1_dist, pack_0_dist, pack_1_dist):
    most_worthy_packet = None
    if pack_0_dist <= robot.battery < pack_1_dist:
        if other_pack_0_dist >= pack_0_dist:
            most_worthy_packet = env.packages[0]
    elif pack_0_dist > robot.battery >= pack_1_dist:
        if other_pack_1_dist >= pack_1_dist:
            most_worthy_packet = env.packages[1]
    elif pack_1_dist <= robot.battery > pack_0_dist:
        if pack_1_dist <= pack_0_dist and pack_1_dist <= other_pack_1_dist:
            most_worthy_packet = env.packages[1]
        elif pack_0_dist <= pack_1_dist and pack_0_dist <= other_pack_0_dist:
            most_worthy_packet = env.packages[0]
        else:
            if pack_0_dist <= pack_1_dist:
                most_worthy_packet = env.packages[0]
            else:
                most_worthy_packet = env.packages[1]
    return most_worthy_packet


# finding the most worthy package that agent should pick to get the maximal credit
def get_worthy_package(env, robot):
    best_package = None
    if len(env.packages) == 2:
        other_robot = env.get_robot((robot + 1) % 2)
        other_pack_0_dist = manhattan_distance(other_robot.position, env.packages[0].position)
        other_pack_1_dist = manhattan_distance(other_robot.position, env.packages[1].position)
        pack_0_dist = manhattan_distance(robot.position, env.packages[0].position)
        pack_1_dist = manhattan_distance(robot.position, env.packages[1].position)

        best_package = decide_best_robot(env, robot, other_pack_0_dist, other_pack_1_dist, pack_0_dist, pack_1_dist)

    else:  # there is one package
        if manhattan_distance(robot.position, env.packages[0].position) < robot.battery:
            best_package = env.packages[0]

    return best_package


#######################################################################################################################
# TODO: section a : 3
def smart_heuristic(env: WarehouseEnv, robot_id: int):
    robot = env.get_robot(robot_id)

    # we have package in the robot but bot enough battery
    if env.robot_is_occupied(robot_id) and not battery_sufficient(robot):  # have to go to charging station
        # deciding which one to go to
        station_1_dist = manhattan_distance(robot.position, env.charge_stations[1].position)
        station_0_dist = manhattan_distance(robot.position, env.charge_stations[0].position)
        closest_station = env.charge_stations[0] if station_0_dist < station_1_dist else env.charge_stations[1]
        dist_to_station = manhattan_distance(robot.position, closest_station.position)
        destination_total_credit = 0
        destination_total_battery = robot.credit + robot.battery - dist_to_station
        return 0.6 * destination_total_credit + 0.4 * destination_total_battery

    # we have package in the robot and enough battery
    elif env.robot_is_occupied(robot_id) and battery_sufficient(robot):
        N = manhattan_distance(robot.package.position, robot.package.destination)
        destination_total_credit = robot.credit + 2 * N
        destination_total_battery = robot.battery - manhattan_distance(robot.position, robot.package.destination)
        return 0.6 * destination_total_credit + 0.4 * destination_total_battery
    # we dont have package
    else:
        worthy_package = get_worthy_package(env, robot)

        if worthy_package is None:  # battery is too low
            station_1_dist = manhattan_distance(robot.position, env.charge_stations[1].position)
            station_0_dist = manhattan_distance(robot.position, env.charge_stations[0].position)
            closest_station = env.charge_stations[0] if station_0_dist < station_1_dist else env.charge_stations[1]
            dist_to_station = manhattan_distance(robot.position, closest_station.position)
            destination_total_credit = 0
            destination_total_battery = robot.credit + robot.battery - dist_to_station
            return 0.6 * destination_total_credit + 0.4 * destination_total_battery

        else:
            N = manhattan_distance(worthy_package.position, worthy_package.destination)
            destination_total_credit = robot.credit + 2 * N
            destination_total_battery = robot.battery - N - manhattan_distance(robot.position,
                                                                               worthy_package.position)
            return 0.6 * destination_total_credit + 0.4 * destination_total_battery


######################################################################################################################
class AgentGreedyImproved(AgentGreedy):
    # TODO: section a : 3
    def run_step(self, env: WarehouseEnv, robot_id, time_limit):
        operators = env.get_legal_operators(robot_id)
        children = [env.clone() for _ in operators]
        for child, op in zip(children, operators):
            child.apply_operator(robot_id, op)
        children_heuristics = [self.heuristic(child, robot_id) for child in children]
        max_heuristic = max(children_heuristics)
        index_selected = children_heuristics.index(max_heuristic)
        return operators[index_selected]

    def heuristic(self, env: WarehouseEnv, robot_id: int):
        return smart_heuristic(env, robot_id)


#######################################################################################################################

class AgentMinimax(Agent):
    # TODO: section b: 1


    def heuristic(self, env: WarehouseEnv, robot_id: int):
        return smart_heuristic(env, robot_id)

    def RB_minimax(self, env: WarehouseEnv, agent_id, turn, time_left, Depth):
        start_time , index_selected , operators = time() , -1 , env.get_legal_operators(turn)
        is_over, result = game_is_over(env, agent_id)
        if is_over:
            return (result, operators[index_selected])
        if (env.done() or Depth == 0) :
            return (self.heuristic(env, agent_id), operators[index_selected])

        children = [env.clone() for _ in operators]
        for child, operation in zip(children, operators):
            child.apply_operator(turn, operation)

        #min vertice
        if (turn == agent_id) :
            curr_maximum = -INF
            for succ in range(len(children)):
                time_left = time_left - (time() - start_time)
                if time_left <= MINIMUM_TIME:
                    break
                c = children[succ]
                value,tmp = self.RB_minimax(c, agent_id, 1 - agent_id, time_left, Depth - 1)
                if value >= curr_maximum:
                    curr_maximum = value
                    index_selected = succ
            return (curr_maximum, operators[index_selected])

        # max vertice
        else:
            curr_minimum = INF
            for succ in range(len(children)):
                curr_time_left = time_left - (time() - start_time)
                if time_left <= MINIMUM_TIME:
                    break
                c = children[succ]
                value, tmp = self.RB_minimax(c, agent_id, agent_id, curr_time_left, Depth - 1)

                if (curr_minimum <= value):
                    curr_minimum = curr_minimum
                else:
                    curr_minimum = value

                time_left = time_left - (time() - start_time)
            return (curr_minimum, operators[index_selected])

    def run_step(self, env: WarehouseEnv, agent_id: int, time_limit):
        depth , start_time , time_left = 1 , time() , time_limit
        current_maximum , current_operation = -INF , None

        while True:
            if time_left <= MINIMUM_TIME:
                break
            value, operation = self.RB_minimax(env, agent_id, agent_id, time_left, depth)
            if value >= current_maximum:
                current_maximum , current_operation = value , operation
            time_left = time_limit - (time() - start_time)
            depth += 1
        return current_operation


#######################################################################################################################

class OutOfTime(Exception):
    pass


#######################################################################################################################
class AgentAlphaBeta(Agent):
    # TODO: section c : 1
    def heuristic(self, env: WarehouseEnv, robot_id: int):
        return smart_heuristic(env, robot_id)

    def game_is_over(self, env, robot_id):
        robot = env.get_robot(robot_id)
        other_robot = env.get_robot((robot_id + 1) % 2)
        if robot.battery == 0 and other_robot.battery == 0 and other_robot.credit > robot.credit:
            return -INF, True

        elif robot.battery == 0 and other_robot.battery == 0 and robot.credit > other_robot.credit:
            return INF, True

        return 0, False

    def alpha_beta_limited_time(self, env, end_time, robot_id, turn, depth, beta, alpha):
        operation_index = -1
        if end_time <= time():
            raise OutOfTime()

        heuristic_res, is_over = self.game_is_over(env, robot_id)
        operators = env.get_legal_operators(turn)

        if is_over:
            return (operators[operation_index], heuristic_res)
        if env.done() or depth <= 0:
            return (operators[operation_index], self.heuristic(env, robot_id))

        children = [env.clone() for _ in operators]
        for child, op in zip(children, operators):
            child.apply_operator(turn, op)

        if turn != robot_id:
            curr_min = float('inf')
            children_num = len(children)
            for i in range(children_num):
                _, curr_val = self.alpha_beta_limited_time(children[i], end_time, robot_id, robot_id, depth - 1, beta,
                                                           alpha)
                if curr_min > curr_val:
                    curr_min = min(curr_val, curr_min)
                    operation_index = i
                beta = min(curr_min, beta)
                if alpha >= curr_min:
                    return (operators[operation_index], float('-inf'))
            return (operators[operation_index], curr_min)
        else:
            maximum_val = float('-inf')
            children_num = len(children)
            for i in range(children_num):
                _, curr_val = self.alpha_beta_limited_time(children[i], end_time, robot_id, 1 - robot_id, depth - 1,
                                                           beta, alpha)
                if maximum_val < curr_val:
                    operation_index = i
                    maximum_val = max(maximum_val, curr_val)
                alpha = max(maximum_val, alpha)
                if beta <= maximum_val:
                    return (operators[i], float('inf'))
            return (operators[operation_index], maximum_val)

    def run_step(self, env: WarehouseEnv, robot_id, time_limit):
        operator_curr = None
        depth = 0
        max_curr = float('-inf')
        end_time = time_limit + time() - EPSILON_LIMIT

        while True:
            try:
                if end_time <= time():
                    raise OutOfTime()
                operation, value = self.alpha_beta_limited_time(env, end_time, robot_id, robot_id, depth, float('inf'),
                                                                float('-inf'))
                depth = depth + 1
                if max_curr <= value:
                    max_curr, operator_curr = value, operation

            except OutOfTime:
                return operator_curr


#######################################################################################################################


class AgentExpectimax(Agent):
    # TODO: Section D - Task 1
    def is_game_over(self, env, agent_id):
        robot = env.get_robot(agent_id)
        opponent_robot = env.get_robot(1 - agent_id)

        if robot.battery == 0 and opponent_robot.battery == 0:
            if robot.credit > opponent_robot.credit:
                return True, 100
            elif robot.credit < opponent_robot.credit:
                return True, -100

        return False, 0

    def get_probabilities(self, operators, env: WarehouseEnv, agent_id, turn):
        special_ops = []
        operators = env.get_legal_operators(turn)
        children = [env.clone() for _ in operators]

        for op in operators:
            if op == 'move north':
                tmp = env.get_robot(agent_id)
                position = tmp.position[1] * 5 + tmp.position[0] - 5
                if is_charging_station(env, position):
                    special_ops.append(op)
            elif op == 'move south':
                tmp = env.get_robot(agent_id)
                position = tmp.position[1] * 5 + tmp.position[0] + 5
                if is_charging_station(env, position):
                    special_ops.append(op)
            elif op == 'move east':
                tmp = env.get_robot(agent_id)
                position = tmp.position[1] * 5 + tmp.position[0] + 1
                if is_charging_station(env, position):
                    special_ops.append(op)
            elif op == 'move west':
                tmp = env.get_robot(agent_id)
                position = tmp.position[1] * 5 + tmp.position[0] - 1
                if is_charging_station(env, position):
                    special_ops.append(op)

        x = 1 / len(operators)
        probabilities = []

        for op in operators:
            if op in special_ops:
                probabilities.append(2 * x)
            else:
                probabilities.append(x)

        return probabilities

    def evaluate(self, env: WarehouseEnv, robot_id: int):
        return smart_heuristic(env, robot_id)

    def run_step(self, env: WarehouseEnv, agent_id, time_limit):
        depth , start_time , time_left = 0 , time() , time_limit
        current_max , current_operator = -INF , None

        while True:
            if time_left <= MINIMUM_TIME:
                break

            value, operator = self.expectimax(env, agent_id, agent_id, time_left, depth)

            if value >= current_max:
                current_max , current_operator = value , operator

            time_left = time_limit - (time() - start_time)
            depth += 1

        return current_operator

    def expectimax(self, env: WarehouseEnv, agent_id, turn, time_left, depth):
        start_time , selected_index , operators = time() ,-1 , env.get_legal_operators(turn)
        children = [env.clone() for _ in operators]
        is_game_over, result = self.is_game_over(env, agent_id)

        if is_game_over:
            return result, operators[selected_index]

        if env.done() or depth == 0:
            return self.evaluate(env, agent_id), operators[selected_index]

        for child, op in zip(children, operators):
            child.apply_operator(turn, op)

        if turn == 1 - agent_id:
            probabilities = self.get_probabilities(operators, env, agent_id, turn)
            children_values = []

            for i in range(len(children)):
                child = children[i]
                time_left = time_left - (time() - start_time)

                if time_left <= MINIMUM_TIME:
                    break

                value, _ = self.expectimax(child, agent_id, 1 - agent_id, time_left, depth - 1)
                children_values.append(probabilities[i] * value)

            return sum(children_values), selected_index
        else:
            current_max = float('-inf')

            for i in range(len(children)):
                child = children[i]
                time_left = time_left - (time() - start_time)

                if time_left <= MINIMUM_TIME:
                    break

                value, _ = self.expectimax(child, agent_id, 1 - agent_id, time_left, depth - 1)

                if value >= current_max:
                    current_max = value
                    selected_index = i

            return current_max, operators[selected_index]


#######################################################################################################################

# here you can check specific paths to get to know the environment
class AgentHardCoded(Agent):
    def __init__(self):
        self.step = 0
        # specifiy the path you want to check - if a move is illegal - the agent will choose a random move
        self.trajectory = ["move north", "move east", "move north", "move north", "pick_up", "move east", "move east",
                           "move south", "move south", "move south", "move south", "drop_off"]

    def run_step(self, env: WarehouseEnv, robot_id, time_limit):
        if self.step == len(self.trajectory):
            return self.run_random_step(env, robot_id, time_limit)
        else:
            op = self.trajectory[self.step]
            if op not in env.get_legal_operators(robot_id):
                op = self.run_random_step(env, robot_id, time_limit)
            self.step += 1
            return op

    def run_random_step(self, env: WarehouseEnv, robot_id, time_limit):
        operators, _ = self.successors(env, robot_id)

        return random.choice(operators)
