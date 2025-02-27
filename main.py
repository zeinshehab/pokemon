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


def get_local_pokemon_positions(grid, pos):
    pokemon_positions = []
    
    start_r, start_c = get_search_start(pos)
    
    for i in range(start_r, start_r + SEARCH_GRID_DIM[0]):
        for j in range(start_c, start_c + SEARCH_GRID_DIM[1]):
            if is_pokemon(grid, (i, j)):
                pokemon_positions.append((i, j))

    return pokemon_positions

distances = {}

def find_optimal_path_alpha_helper(grid, start, path, depth, alpha, densities):    
    current_score = len(path)
    pokemons_left = NUM_POKEMONS - current_score
    
    if pokemons_left <= 0:
        return path
    
    
    # try ceil and floor and nothing
    upper_bound = min(current_score + (math.ceil(depth/AVG_MOVES_FOR_POKEMON)),
                      current_score + (math.ceil(pokemons_left/AVG_MOVES_FOR_POKEMON)))
    
    if depth <= 0:
        return path
    
    if upper_bound <= alpha:
        return path
    
    local_pokemons = get_local_pokemon_positions(grid, start)
    
    # sort pokemons by (number of local pokemons around)
    # local_pokemons.sort(key=lambda p: densities[p], reverse=True)    
    
    # sort pokemons by distance and if its not in the distances dict
    # calculate the distance and store it
    for pokemon in local_pokemons:
        if (start, pokemon) not in distances:
            distances[(start, pokemon)] = distance(start, pokemon)
    
    order_local_positions(local_pokemons, densities, start)
    
    # local_pokemons.sort(key=lambda p: distances[(start, p)])
    # random.shuffle(local_pokemons)
    
    max_score = current_score
    best_path = path
    
    for pokemon in local_pokemons:
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
            
        if (upper_bound <= alpha):
            break
            
    return best_path
 
 
def find_optimal_path_alpha(grid, start, depth, alpha, densities):
    return find_optimal_path_alpha_helper(grid, start, [start], depth, alpha, densities)
 

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
        density[pokemon] = len(local_pokemons)
        
    return density


def order_start_positions(positions, densities, closest_distances):
    positions.sort(key=lambda x: densities[x] - closest_distances[x], reverse=True)


def order_local_positions(positions, densities, start):
    positions.sort(key=lambda x: densities[x] - distance(start, x), reverse=True)
        

def reoder_start_positions(positions, best_start):
    positions.sort(key=lambda x: distance(best_start, x))


def main():
    global WIDTH, HEIGHT, SEARCH_GRID_DIM, AVG_MOVES_FOR_POKEMON, NUM_POKEMONS

    append = False

    old_best_path = [(70, 21), (70, 22), (69, 22), (68, 22), (67, 22), (66, 22), (66, 21), (67, 21), (68, 21), (68, 20), (68, 19), (67, 19), (67, 18), (66, 18), (66, 17), (66, 16), (65, 16), (64, 16), (63, 16), (62, 16), (62, 17), (62, 18), (62, 19), (61, 19), (61, 20), (60, 20), (60, 19), (59, 19), (58, 19), (57, 19), (57, 20), (57, 21), (56, 21), (56, 20), (56, 19), (55, 19), (54, 19), (54, 20), (53, 20), (53, 19), (53, 18), (53, 17), (54, 17), (54, 16), (54, 15), (55, 15), (55, 14), (55, 13), (55, 12), (54, 12), (54, 11)]


    acc = Accuracy.ROUND

    SEARCH_GRID_DIM =  (int(sys.argv[1]), int(sys.argv[2]))
    MAX_NUM_MOVES = int(sys.argv[3])
    grid, dimensions = read_data(sys.argv[4])
    HEIGHT, WIDTH, workers = dimensions
                
    if SEARCH_GRID_DIM[0] > HEIGHT:
        SEARCH_GRID_DIM = (HEIGHT, SEARCH_GRID_DIM[1])
        
    if SEARCH_GRID_DIM[1] > WIDTH:
        SEARCH_GRID_DIM = (SEARCH_GRID_DIM[0], WIDTH)

    if append:
        MAX_NUM_MOVES = MAX_NUM_MOVES - len(pokemons_to_moves(old_best_path[0], old_best_path))  
        
        for p in old_best_path[:len(old_best_path)-1]:
            set_at(grid, p, ".")

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
            
    if acc == Accuracy.HIGH:
        # soft pruning. Gives slow/decently fast solution better than theoretical score
        AVG_MOVES_FOR_POKEMON = math.floor(average_moves)
        print("Using HIGH accuracy mode")
    elif acc == Accuracy.MEDIUM:
        # normal pruning. Give fast solution around the theoretical score
        AVG_MOVES_FOR_POKEMON = average_moves
        print("Using MEDIUM accuracy mode")
    elif acc == Accuracy.LOW:
        # Harsh pruning. Give fastest solution but worse than theoretical score
        AVG_MOVES_FOR_POKEMON = math.ceil(average_moves)
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
            
    theoretical_average = MAX_NUM_MOVES / average_moves + 1 # + 1 for starting at pokemon without making a move
    theoritical_optimal = MAX_NUM_MOVES / math.floor(AVG_MOVES_FOR_POKEMON) + 1 # + 1 for starting at pokemon without making a move
    
    print(f"\nTheoretical average estimate: {theoretical_average:.2f}")
    print(f"Theoretical optimal estimate: {theoritical_optimal:.2f}")
            
    print("\n--------------------------------------------------------\n")    
    print(f"Finding fast optimal path...\n")


    # print(pokemons_to_moves((70, 21), best_path))
    # print(f"Score: {score_path(grid, pokemons_to_moves((70, 21), best_path), (70, 21))}")
    
    # sys.exit(1)
    
    best_start = (70, 21)
    if append:
        best_start = old_best_path[-1]
        start_positions = [best_start]
    else:
        start_positions = global_pokemon_positions
        
        # currently we order the positions by a balance between density and distance to closest pokemon
        order_start_positions(start_positions, densities, closest_distances)    

        # this reorders the start positions based on proximity to the old best start
        reoder_start_positions(start_positions, best_start)        
    
    best_path = []
    best_score = 0
    
    # start_positions = [(0, 0)]
    
    # sort positions by distance to closest pokemon. aka which one has the nearest pokemon around
    # start_positions.sort(key=lambda p: closest_distances[p])
    
    # sort postions by distance to center of the grid
    # start_positions.sort(key=lambda p: distance(p, (HEIGHT//2, WIDTH//2)))
    
    # sort positions relative to median value of density
    # start_positions.sort(key=lambda p: abs(densities[p] - avg_density))
    
    # randomly shuffle the start positions
    # random.shuffle(start_positions)

    scores = {}
    
    prune = bool(int(sys.argv[5]))

    print(f"Exploring {len(start_positions)} start positions...\n")
    for i in tqdm(range(NUM_POKEMONS), desc=f"Processing", unit="pos"):
        if len(start_positions) == 0:
            break
        
        start = start_positions[0]
        
        set_at(grid, start, ".")
        
        start_time = time.perf_counter()
        
        if prune:
            fast_path = find_optimal_path_alpha(grid, start, MAX_NUM_MOVES, best_score, densities)
        else:
            fast_path = find_optimal_path_alpha(grid, start, MAX_NUM_MOVES, 0, densities)
        end_time = time.perf_counter()
        fast_time = end_time - start_time
        
        score = len(fast_path)
        
        scores[start] = score
        
        # tqdm.write(f"Start: {start} | Density: {densities[start]} | Time: {fast_time:.2f} | Score: {score}")
            
        start_positions.pop(0)
                
        if score > best_score:
            reoder_start_positions(start_positions, best_start)
            
            best_score = score
            best_start = start
            best_path = fast_path
            
        set_at(grid, start, "X")
        
        if best_score == theoritical_optimal:
            break
                
    threshold = theoretical_average - 5
    with open("output.txt", "w+") as f:
        best = []
        for start, score in scores.items():
            if score > threshold:
                best.append(start)
        f.write(" , ".join([str(x) for x in best]))
    
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
    print(f"Actual alpha score: {score_path(grid, best_moves, best_start)}")
    
    if append:
        for p in old_best_path:
            set_at(grid, p, "X")
        
        print("\n--------------------------------------------------------\n")
        new_path = old_best_path + best_path[1:]    
        new_moves = pokemons_to_moves(old_best_path[0], new_path)
        print(new_path)
        print(new_moves)
        print(f"New moves used: {len(new_moves)}")
        print(f"Score: {score_path(grid, new_moves, old_best_path[0])}")
        
        

if __name__ == '__main__':
    main()