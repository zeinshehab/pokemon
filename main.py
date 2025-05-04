from collections import defaultdict, deque
import math
import random
from tqdm import tqdm

import time
import sys


directions = {
    "^": (-1, 0),
    "v": (1, 0),
    ">": (0, 1),
    "<": (0, -1)
}

distances = {}

MAX_NUM_MOVES = 100
WIDTH, HEIGHT = 0,0
AVG_MOVES_FOR_POKEMON = 1
SEARCH_GRID_DIM = (5, 5)
NUM_POKEMONS = 0

class Accuracy:
    LOW = 0
    MEDIUM = 1
    HIGH = 2
    ROUND = 3
    
    
def print_grid(grid):
    for row in grid:
        print("".join(row))


def get_at(grid, pos):
    return grid[pos[0]][pos[1]]


def set_at(grid, pos, val):
    grid[pos[0]][pos[1]] = val
    return grid


def out_of_bounds(grid, pos):
    if pos[0] < 0 or pos[0] >= HEIGHT or pos[1] < 0 or pos[1] >= WIDTH:
        return True
    return False


def is_pokemon(grid, pos):
    return get_at(grid, pos) == "X" or get_at(grid, pos) == "x"


def update_pos(pos, direction):
    return (pos[0] + direction[0], pos[1] + direction[1])


def read_data(file):
    with open(file, "r") as data_file:
        lines = data_file.read().splitlines()
        dimensions = lines[0].split(" ")
        dimensions = [int(x) for x in dimensions]
        grid = lines[1:]

        grid = [list(x) for x in grid]

        return grid, dimensions
    return None, None


def get_search_start(pos):
    r, c = pos
    half_dim_y = SEARCH_GRID_DIM[0] // 2
    half_dim_x = SEARCH_GRID_DIM[1] // 2

    # Clamp the starting position to stay within bounds
    start_r = max(0, min(r - half_dim_y, HEIGHT - SEARCH_GRID_DIM[0]))
    start_c = max(0, min(c - half_dim_x, WIDTH - SEARCH_GRID_DIM[1]))

    return start_r, start_c


def distance(pos_a, pos_b):
    return abs(pos_a[0] - pos_b[0]) + abs(pos_a[1] - pos_b[1])


def get_global_pokemon_positions(grid):
    # global_pokemon_positions = set()
    global_pokemon_positions = []

    for i in tqdm(range(HEIGHT)):
        for j in range(WIDTH):
            if is_pokemon(grid, (i, j)):
                # global_pokemon_positions.add((i, j))
                global_pokemon_positions.append((i, j))
    return global_pokemon_positions


def filter_local_pokemons(current, local_pokemons):
    non_dominated = []
    for p in local_pokemons:
        dominated = False
        for q in local_pokemons:
            if p == q:
                continue
            
            # if distances not in distances dict, add it
            if (current, p) not in distances:
                distances[(current, p)] = distance(current, p)
            if (current, q) not in distances:
                distances[(current, q)] = distance(current, q)
            if (q, p) not in distances:
                distances[(q, p)] = distance(q, p)
                
            # If Q is closer than P and lies exactly on an optimal Manhattan path from current to P
            if distances[(current, q)] < distances[(current, p)] and \
               distances[(current, q)] + distances[(q, p)] == distances[(current, p)]:
                dominated = True
                break
            
        if not dominated:
            non_dominated.append(p)
    return non_dominated


def get_local_pokemon_positions(grid, pos):
    pokemon_positions = []
    
    start_r, start_c = get_search_start(pos)
    
    for i in range(start_r, start_r + SEARCH_GRID_DIM[0]):
        for j in range(start_c, start_c + SEARCH_GRID_DIM[1]):
            if is_pokemon(grid, (i, j)):
                pokemon_positions.append((i, j))

    return pokemon_positions


def find_optimal_path_alpha_helper(grid, start, path, depth, alpha, densities):    
    current_score = len(path)
    pokemons_left = NUM_POKEMONS - current_score
    
    if pokemons_left <= 0:
        return path
    
    upper_bound = min(current_score + (math.ceil(depth/AVG_MOVES_FOR_POKEMON)),
                      current_score + (math.ceil(pokemons_left/AVG_MOVES_FOR_POKEMON)))
    
    if depth <= 0:
        return path
    
    if upper_bound <= alpha:
        return path
    
    local_pokemons = get_local_pokemon_positions(grid, start)

    filtered_pokemons = filter_local_pokemons(start, local_pokemons)
    filtered_pokemons = local_pokemons
        
    # order_local_positions(filtered_pokemons, densities, start)
    
    # filtered_pokemons.sort(key=lambda p: distances[(start, p)])
    # random.shuffle(filtered_pokemons)
    
    max_score = current_score
    best_path = path
    
    for pokemon in filtered_pokemons:
        if (start, pokemon) not in distances:
            distances[(start, pokemon)] = distance(start, pokemon)
        dist = distances[(start, pokemon)]
        
        if depth - dist < 0:
            continue

        set_at(grid, pokemon, ".")  # Remove from grid
        new_path = find_optimal_path_alpha_helper(grid, pokemon, path + [pokemon], depth - dist, alpha, densities)
        set_at(grid, pokemon, "X")  # Restore to grid

        # score = (depth - dist) + len(new_path)
        score = len(new_path)
        if score > max_score:
            max_score = score
            best_path = new_path
            
        if max_score > alpha:
            alpha = max_score
            
        if upper_bound <= alpha:
            break
            
    return best_path
 
 
def find_optimal_path_alpha(grid, start, depth, alpha, densities):
    return find_optimal_path_alpha_helper(grid, start, [start], depth, alpha, densities)
 

def iddfs_find_best_path(grid, start, alpha, densities, max_depth=100):
    best_path = []
    alpha = 0
    
    for depth in range(5, max_depth + 1, 5):  # try every 5 moves
        path = find_optimal_path_alpha_helper(grid, start, [], depth, alpha, densities)
        if len(path) > len(best_path):
            best_path = path
            alpha = len(best_path)
        # Optional: break early if you're hitting max pokemons
        if alpha == NUM_POKEMONS:
            break
    return best_path


def pokemons_to_moves(start, pokemons):
    moves = []
    
    for pokemon in pokemons:
        vert_dist = start[0] - pokemon[0]
        if vert_dist > 0:
            moves += ["^"] * abs(vert_dist)
        elif vert_dist < 0:
            moves += ["v"] * abs(vert_dist)
            
        horiz_dist = start[1] - pokemon[1]
        if horiz_dist > 0:
            moves += ["<"] * abs(horiz_dist)
        elif horiz_dist < 0:
            moves += [">"] * abs(horiz_dist)
        start = pokemon
    return "".join(moves)
 

def score_path(grid, path, pos):
    visited = set()
    score = 0

    # Collect Pokémon at the start
    if is_pokemon(grid, pos):
        visited.add(pos)
        score += 1

    for step in path:
        direction = directions[step]
        pos = update_pos(pos, direction)

        if out_of_bounds(grid, pos):
            return None

        if is_pokemon(grid, pos) and pos not in visited:
            visited.add(pos)
            score += 1

    return score


def distances_closest_pokemon(grid, global_pokemon_positions):
    distances = defaultdict(lambda: 9999999)
    
    for pokemon in global_pokemon_positions:
        for other in get_local_pokemon_positions(grid, pokemon):
            if pokemon == other:
                continue
            dist = distance(pokemon, other)
            if dist < distances[pokemon]:
                distances[pokemon] = dist
                
    return distances


def pokemon_densities(grid, global_pokemon_positions):
    density = defaultdict(int)
    
    for pokemon in global_pokemon_positions:
        local_pokemons = get_local_pokemon_positions(grid, pokemon)

        # overhead is not worth it
        # local_pokemons = filter_local_pokemons(pokemon, local_pokemons)
        density[pokemon] = len(local_pokemons)
        
    return density


def greedy_start_scores(grid, global_pokemon_positions, moves):
    def distances_closest_pokemon(grid, global_pokemon_positions):
        distances = defaultdict(lambda: [])
        
        for pokemon in global_pokemon_positions:
            for other in get_local_pokemon_positions(grid, pokemon):
                if pokemon == other:
                    continue
                dist = distance(pokemon, other)
                distances[pokemon].append((other, dist))
                # if dist < distances[pokemon][1]:
                #     distances[pokemon] = (other, dist)
            distances[pokemon].sort(key=lambda x: x[1])
        return distances
    
    
    starts = {}
    all_paths = []
    # Try each Pokémon as a starting point.
    closest_distances = distances_closest_pokemon(grid, global_pokemon_positions)
    
    for start in closest_distances.keys():
        moves_left = moves
        path = [start]
        visited = {start}
        current = start

        while current in closest_distances:
            candidates = closest_distances[current]
            next_candidate = None

            # Look for the next candidate that hasn't been visited and is affordable.
            for candidate, cost in candidates:
                if candidate not in visited and moves_left >= cost:
                    next_candidate = (candidate, cost)
                    break
                    
            if not next_candidate:
                break
                
            candidate, move_cost = next_candidate
            moves_left -= move_cost
            path.append(candidate)
            visited.add(candidate)
            current = candidate
            
            if moves_left <= 0:
                break
            
        all_paths.append(path)
    
    # Sort the paths by length in descending order.
    all_paths.sort(key=lambda p: len(p), reverse=True)
    # Return the top 5 best paths.
    for path in all_paths:
        starts[path[0]] = len(path)
    
    # all_paths = [x[0] for x in all_paths]
    # top_5 = all_paths[:5]
    return starts


def order_start_positions(grid, positions, densities, closest_distances):
    # start_scores = greedy_start_scores(grid, positions, MAX_NUM_MOVES)
    # median = sum(start_scores.values()) / len(start_scores)
    
    # random.shuffle(positions)
    
    # positions.sort(key=lambda x: (abs(start_scores[x]-median)) + densities[x] + closest_distances[x], reverse=False)
    # positions.sort(key=lambda x: densities[x] / start_scores[x], reverse=True)
    positions.sort(key=lambda x: densities[x] + closest_distances[x], reverse=False)


def order_local_positions(positions, densities, start):
    # positions.sort(key=lambda x: densities[x] - distance(start, x), reverse=True)
    positions.sort(key=lambda x: distance(start, x))
        

def reoder_start_positions(positions, best_start):
    positions.sort(key=lambda x: distance(best_start, x))


def main():
    global WIDTH, HEIGHT, SEARCH_GRID_DIM, AVG_MOVES_FOR_POKEMON, NUM_POKEMONS

    SEARCH_GRID_DIM =  (int(sys.argv[1]), int(sys.argv[2]))
    MAX_NUM_MOVES = int(sys.argv[3])
    grid, dimensions = read_data(sys.argv[4])
    prune = bool(int(sys.argv[5]))
    HEIGHT, WIDTH, workers = dimensions
                
    if SEARCH_GRID_DIM[0] > HEIGHT:
        SEARCH_GRID_DIM = (HEIGHT, SEARCH_GRID_DIM[1])
        
    if SEARCH_GRID_DIM[1] > WIDTH:
        SEARCH_GRID_DIM = (SEARCH_GRID_DIM[0], WIDTH)

    print("[*] Finding pokemon positions...")
    global_pokemon_positions = get_global_pokemon_positions(grid)
                                
    NUM_POKEMONS = len(global_pokemon_positions)                                
                                
    print("[*] Calculating avg moves and avg density ...")

    start_time = time.perf_counter()

    closest_distances = distances_closest_pokemon(grid, global_pokemon_positions)
    average_moves = sum(closest_distances.values()) / len(closest_distances)   
    densities = pokemon_densities(grid, global_pokemon_positions)
    avg_density = sum(densities.values()) / len(densities) 
                
    end_time = time.perf_counter()
    dist_time = end_time - start_time
    
    print("\n--------------------------------------------------------\n")
    print(f"Time taken to calculate averages: {dist_time:.2f}s\n")
            
    # acc = Accuracy.MEDIUM
    acc = int(sys.argv[6])
    
    if acc == Accuracy.HIGH:
        # soft pruning. Gives slow/decently fast solution better than theoretical score
        AVG_MOVES_FOR_POKEMON = 1
        print("Using HIGH accuracy mode")
    elif acc == Accuracy.MEDIUM:
        # normal pruning. Give fast solution around the theoretical score
        AVG_MOVES_FOR_POKEMON = average_moves
        print("Using MEDIUM accuracy mode")
    elif acc == Accuracy.LOW:
        # Harsh pruning. Give fastest solution but worse than theoretical score
        AVG_MOVES_FOR_POKEMON = math.floor(average_moves) + 1
        print("Using LOW accuracy mode")
    else:
        AVG_MOVES_FOR_POKEMON = round(average_moves)
        print("Using ROUND accuracy mode")
            
    # this helps to set the upper bound for the alpha score
    # based on how many moves on average you need to reach a pokemon
    print("Average number of moves to reach a pokemon: {:.2f} | Using: {:.2f}".format(average_moves, AVG_MOVES_FOR_POKEMON))
    
    # this helps me to pick what is the best search dimension 
    # based on how full/sparse the grid is
    # i want a decent number of pokemons (approx 15-20) in the search area to explore enough candidates 
    # to make it more probabilistc that i explored the optimal solution path
    # for a more sparse grid i increase the search dimension to capture more pokemons around me
    print("Average number of local pokemons for SEARCH_DIM {}: {:.2f}".format(SEARCH_GRID_DIM, avg_density))
            
    theoretical_average = min(MAX_NUM_MOVES, NUM_POKEMONS) / average_moves + 1 # + 1 for starting at pokemon without making a move
    theoritical_optimal = min(MAX_NUM_MOVES / math.floor(average_moves) + 1, NUM_POKEMONS)  # + 1 for starting at pokemon without making a move
    
    print(f"\nTheoretical average estimate: {theoretical_average:.2f}")
    print(f"Theoretical optimal estimate: {theoritical_optimal:.2f}")
            
    g_scores = greedy_start_scores(grid, global_pokemon_positions, MAX_NUM_MOVES)
    best_greedy = max(g_scores, key=g_scores.get)
    best_greedy_score = g_scores[best_greedy]
    print(f"Best greedy start: {best_greedy} | Score: {best_greedy_score}")  
    
    print("\n--------------------------------------------------------\n")    
    print(f"Finding fast optimal path...\n")

    start_positions = global_pokemon_positions   

    # best_start = (278, 65)
    # best_start = (49, 126)
    # best_start = (62, 149)
    # best_start = (44, 14)
    # best_start = (70, 21)
    best_start = None 
    best_path = []
    best_score = 0
    if acc == Accuracy.HIGH:
        best_score = theoretical_average
    
    # path = ">^^^^^<vvv<<^^<<<^^^^>>vv>^^^>^<^^<^>>v>^>^<<<^^>>^^<v<<<v<<v<<^<vv<^^^<v<^^<^^>>^^^<<^>>>^^>>>^<^>^"
    # path = "vvv>v<^^>>>^>>>>>>>>vv>>>>>v<v<<<v<<<<<^^>v<<<v<vv<vv>vvvv^>>vvv<>>>>vv<<v>>>>>>^>>^<<<<^^^^^v>>^>>>"
    # path = "v<^^^<<<vvvvvvvvv^>>vv>>vvv<v<<vv>v>>vvv<<vv>>vvvvv<<v>vvv<>>vvv<<vvv>>v<<<<vvvv>>>vv<vv<vv<>>>v<vv>"
    
    # print(score_path(grid, path, (0, 3)))
    # sys.exit(1)
    g_scores = greedy_start_scores(grid, start_positions, MAX_NUM_MOVES)
    avg_g_score = sum(g_scores.values()) / len(g_scores)
    
    def pos_statistics(pos):
        print(f"Statistics {pos}:")
        g = g_scores[pos]
        d = densities[pos]
        c = closest_distances[pos]
        print(f"Greedy Score: {g} | Average Greedy Score: {avg_g_score:.2f}")
        print(f"Density: {d} | Average Density: {avg_density:.2f}")
        print(f"Closest distance: {c} | Average Closest Distance: {average_moves:.2f}")
        print("--------------------------------------------------------\n")


    # filter start positions. keep only the ones with g score greater than average
    # start_positions = [x for x in start_positions if g_scores[x] >= avg_g_score]
    # start_positions = [x for x in start_positions if densities[x] >= avg_density]
    # start_positions = [x for x in start_positions if closest_distances[x] <= average_moves]

    # currently we order the positions by a balance between density and distance to closest pokemon
    order_start_positions(grid, start_positions, densities, closest_distances)    
    print(f"Best Start after ordering:")
    pos_statistics(start_positions[0])
    

    # this reorders the start positions based on proximity to the old best start
    if best_start is not None:
        pos_statistics(best_start)
        reoder_start_positions(start_positions, best_start)   
    
    # start_positions = [random.choice(start_positions)] # run only one iteration
    
        pos_statistics(start_positions[0])
    
    scores = {}
    
    n = len(start_positions)
    print(f"Exploring {n} start positions...\n")
    for i in tqdm(range(n), desc=f"Processing", unit="pos"):
        if len(start_positions) == 0:
            break
        
        start = start_positions[0]
        
        set_at(grid, start, ".")
        
        start_time = time.perf_counter()
        if prune:
            # fast_path = iddfs_find_best_path(grid, start, best_score, densities, MAX_NUM_MOVES)
            fast_path = find_optimal_path_alpha(grid, start, MAX_NUM_MOVES, best_score, densities)
        else:
            fast_path = find_optimal_path_alpha(grid, start, MAX_NUM_MOVES, 0, densities)
        end_time = time.perf_counter()
        fast_time = end_time - start_time
        
        score = len(fast_path)
        scores[start] = score
        
        start_positions.pop(0)
                
        if score > best_score:
            best_score = score
            best_start = start
            best_path = fast_path
            
            reoder_start_positions(start_positions, best_start)
            
        set_at(grid, start, "X")
        
        if best_score == theoritical_optimal:
            break

    print("\n--------------------------------------------------------\n") 
    print(f"Best start: {best_start}\n")   
    best_moves = pokemons_to_moves(best_start, best_path)
    print(f"{best_path}")
    print(f"Pokemons captured: {len(best_path)}\n")
    print(best_moves)
    print(f"Moves used: {len(best_moves)}")
    print("\n--------------------------------------------------------\n")
    
    print(f"Theoretical average estimate: {theoretical_average:.2f}")
    print(f"Theoretical optimal estimate: {theoritical_optimal:.2f}")
    print(f"Actual score: {score_path(grid, best_moves, best_start)}")


if __name__ == '__main__':
    main()

