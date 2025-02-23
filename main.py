from collections import defaultdict
import math
from tqdm import tqdm

import os
import time
import sys

directions = {
    "^": (-1, 0),
    "v": (1, 0),
    ">": (0, 1),
    "<": (0, -1)
}

REWARD  =  1
PENALTY = 0
MAX_NUM_MOVES = 100
WIDTH, HEIGHT = 0,0
AVG_MOVES_FOR_POKEMON = 1
SEARCH_DIM = 5

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
    return get_at(grid, pos) == "x"


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
    half_dim = SEARCH_DIM // 2

    # Clamp the starting position to stay within bounds
    start_r = max(0, min(r - half_dim, HEIGHT - SEARCH_DIM))
    start_c = max(0, min(c - half_dim, WIDTH - SEARCH_DIM))

    return start_r, start_c


def distance(pos_a, pos_b):
    return abs(pos_a[0] - pos_b[0]) + abs(pos_a[1] - pos_b[1])


def get_global_pokemon_positions(grid, start):
    global_pokemon_positions = set()

    for i in tqdm(range(HEIGHT)):
        for j in range(WIDTH):
            if is_pokemon(grid, (i, j)):
                global_pokemon_positions.add((i, j))
                # global_pokemon_positions.append((i, j))
    return global_pokemon_positions


def get_local_pokemon_positions(grid, pos):
    pokemon_positions = []
    
    start_r, start_c = get_search_start(pos)
    for i in range(start_r, start_r + SEARCH_DIM):
        for j in range(start_c, start_c + SEARCH_DIM):
            if is_pokemon(grid, (i, j)):
                pokemon_positions.append((i, j))

    return pokemon_positions


def find_optimal_path_alpha_helper(grid, start, path, depth, distances, alpha):
    global search_time  # Use global variables
    
    current_score = len(path)
    upper_bound = current_score + (math.ceil(depth/AVG_MOVES_FOR_POKEMON))
    
    if depth <= 0:
        return path
    
    if upper_bound <= alpha:
        return path
    
    local_pokemons = get_local_pokemon_positions(grid, start)
    # local_pokemons.sort(key=lambda p: distances[(start, p)])
    
    max_score = current_score
    best_path = path
    
    for pokemon in local_pokemons:
        dist = distances[(start, pokemon)]
        
        if depth - dist < 0:
            continue

        set_at(grid, pokemon, ".")  # Remove from grid
        new_path = find_optimal_path_alpha_helper(grid, pokemon, path + [pokemon], depth - dist, distances, alpha)
        set_at(grid, pokemon, "x")  # Restore to grid

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
 
 
def find_optimal_path_alpha(grid, start, depth, distances):
    return find_optimal_path_alpha_helper(grid, start, [], depth, distances, 0)


def find_optimal_path_slow_helper(grid, start, path, depth, distances):
    global search_time, explore_time  # Use global variables
    
    if depth <= 0:
        return path

    pokemon_positions = []

    start_r, start_c = get_search_start(start)
    for i in range(start_r, start_r + SEARCH_DIM):
        for j in range(start_c, start_c + SEARCH_DIM):
            if is_pokemon(grid, (i, j)):
                pokemon_positions.append((i, j))
    
    # pokemon_positions.sort(key=lambda p: distances[(start, p)])
    
    max_score = 0
    best_path = path
    
    for pokemon in pokemon_positions:
        dist = distances[(start, pokemon)]
        
        if depth - dist < 0:
            continue

        grid[pokemon[0]][pokemon[1]] = "."  # Remove from grid
        new_path = find_optimal_path_slow_helper(grid, pokemon, path + [pokemon], depth - dist, distances)
        grid[pokemon[0]][pokemon[1]] = "x"  # Restore to grid

        score = (depth - dist) + len(new_path)
        if score > max_score:
            max_score = score
            best_path = new_path
            
    return best_path

def find_optimal_path_slow(grid, start, depth, distances):
    return find_optimal_path_slow_helper(grid, start, [], depth, distances)
 

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
        score += REWARD

    for step in path:
        direction = directions[step]
        pos = update_pos(pos, direction)

        if out_of_bounds(grid, pos):
            return None

        if is_pokemon(grid, pos) and pos not in visited:
            visited.add(pos)
            score += REWARD

    return score
 
 
def clear_screen():
    """ Clears the terminal screen """
    os.system("clear" if os.name == "posix" else "cls")


def print_visual_grid(grid, visited, pos):
    """ Prints the grid with path visualization """
    clear_screen()
    visual_grid = [row[:] for row in grid]  # Copy grid to avoid modifying original

    # Mark visited positions
    for v in visited:
        visual_grid[v[0]][v[1]] = "."

    # Mark current position
    visual_grid[pos[0]][pos[1]] = "P"

    # Print the updated grid
    for row in visual_grid:
        print("".join(row))
    
    time.sleep(0.05)  # Adjust for speed


def visualize_search(grid, path, start):
    """ Simulates pathfinding with real-time visualization """
    pos = start
    visited = set()
    visited.add(pos)

    for step in path:
        direction = directions[step]
        pos = update_pos(pos, direction)
        
        if out_of_bounds(grid, pos):
            break

        visited.add(pos)
        print_visual_grid(grid, visited, pos)

    print("Final Path Found!")
    time.sleep(2)


def calculate_distances(start, global_pokemon_positions):
    distances = {}
    
    pokemons_with_start = global_pokemon_positions
    if start not in global_pokemon_positions:
        pokemons_with_start.add(start)
    
    for pokemon in tqdm(pokemons_with_start):
        for other in pokemons_with_start:
            if pokemon == other and start not in global_pokemon_positions:
                continue
            dist = distance(pokemon, other)
            distances[(pokemon, other)] = dist
            
    return distances


def average_moves_for_pokemon(grid, start, global_pokemon_positions):
    distances = defaultdict(lambda: 9999999)
    
    pokemons_with_start = global_pokemon_positions
    if start not in global_pokemon_positions:
        pokemons_with_start.add(start)
            
    for pokemon in pokemons_with_start:
        for other in get_local_pokemon_positions(grid, pokemon):
            if pokemon == other:
                continue
            dist = distance(pokemon, other)
            if dist < distances[pokemon]:
                distances[pokemon] = dist
                
    distances = distances.values()
    return sum(distances) / len(distances)


def main():
    global WIDTH, HEIGHT, SEARCH_DIM, AVG_MOVES_FOR_POKEMON

    start = (0,0)
    SEARCH_DIM =  int(sys.argv[1])
    MAX_NUM_MOVES = int(sys.argv[2])
    grid, dimensions = read_data(sys.argv[3])
    WIDTH, HEIGHT, workers = dimensions

    if SEARCH_DIM > min(WIDTH, HEIGHT):
        SEARCH_DIM = min(WIDTH, HEIGHT)

    print("[*] Calculating pokemon positions...")
    global_pokemon_positions = get_global_pokemon_positions(grid, start)
                                
    print("[*] Calculating distances...")

    start_time = time.time()
    distances = calculate_distances(start, global_pokemon_positions)

    average_moves = average_moves_for_pokemon(grid, start, global_pokemon_positions)
    print("[*] Average number of moves to reach a pokemon: {:.2f} | Floor: {:.2f}".format(average_moves, math.floor(average_moves)))
            
    end_time = time.time()
    dist_time = end_time - start_time
            
    AVG_MOVES_FOR_POKEMON = math.floor(average_moves)
            
    print("[*] Finding fast optimal path...")

    start_time = time.time()
    fast_path = find_optimal_path_alpha(grid, start, MAX_NUM_MOVES, distances)
    end_time = time.time()
    fast_time = end_time - start_time
    
    moves_fast = pokemons_to_moves(start, fast_path)
    
    # print("[*] Finding slow optimal path...")

    # start_time = time.time()
    # slow_path = find_optimal_path_slow(grid, start, depth, distances)
    # end_time = time.time()
    # slow_time = end_time - start_time
    
    # moves_slow = pokemons_to_moves(start, slow_path)
    # visualize_search(grid, moves_fast, start)

    print(f"Time taken to calculate distance: {dist_time:.2f}s")
    print(f"Time taken to find fast: {fast_time:.2f}s")
    # print(f"Time taken to find slow: {slow_time:.2f}s")


    print(fast_path)
    print(f"Len: {len(fast_path)}")
    print(moves_fast)
    print(f"moves: {len(moves_fast)}\n")

    # print(slow_path)
    # print(f"Len: {len(slow_path)}")
    # print(moves_slow)
    # print(f"moves: {len(moves_slow)}\n")
    
    theoretical_score = MAX_NUM_MOVES / AVG_MOVES_FOR_POKEMON
    
    print(f"Theoretical score:  {theoretical_score:.2f}")
    print(f"Actual alpha score: {score_path(grid, moves_fast, start)}")
    # print(f"Score slow: {score_path(grid, moves_slow, start)}")

if __name__ == '__main__':
    main()