from collections import defaultdict, deque
import heapq
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
MAX_PATH_LEN = 100
WIDTH, HEIGHT = 0,0

SEARCH_DIM = int(sys.argv[1])

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


def get_subgrid(grid, pos):
    r, c = pos
    half_dim = SEARCH_DIM // 2

    # Ensure boundaries stay within the grid
    top = max(0, r - half_dim)
    bottom = min(len(grid), top + SEARCH_DIM)  # Ensures full size if possible
    left = max(0, c - half_dim)
    right = min(len(grid[0]), left + SEARCH_DIM)

    return [row[left:right] for row in grid[top:bottom]]


def get_search_start(pos):
    r, c = pos
    half_dim = SEARCH_DIM // 2

    # Clamp the starting position to stay within bounds
    start_r = max(0, min(r - half_dim, HEIGHT - SEARCH_DIM))
    start_c = max(0, min(c - half_dim, WIDTH - SEARCH_DIM))

    return start_r, start_c


def distance(pos_a, pos_b):
    return abs(pos_a[0] - pos_b[0]) + abs(pos_a[1] - pos_b[1])


# def find_optimal_path_helper(grid, start, path, depth, distances):
#     if depth <= 0:
#         return path

#     pokemon_positions = []

#     start_r, start_c = get_search_start(start)
#     for i in range(start_r, start_r + SEARCH_DIM):
#         for j in range(start_c, start_c + SEARCH_DIM):
#             if is_pokemon(grid, (i, j)):
#                 pokemon_positions.append((i, j))
    
#     pokemon_positions.sort(key=lambda p: distances[(start, p)])
    
#     max_score = 0
#     best_path = path
    
#     for pokemon in pokemon_positions:
#         dist = distances[(start, pokemon)]
        
#         if depth - dist < 0:
#             continue

#         grid[pokemon[0]][pokemon[1]] = "."  # Remove from grid
#         new_path = find_optimal_path_helper(grid, pokemon, path + [pokemon], depth - dist, distances)
#         grid[pokemon[0]][pokemon[1]] = "x"  # Restore to grid

#         score = (depth - dist) + len(new_path)
#         if score > max_score:
#             max_score = score
#             best_path = new_path

#     return best_path


def find_optimal_path_helper(grid, start, path, depth):
    global search_time, explore_time  # Use global variables

    if depth <= 0:
        return path

    # Measure the time spent searching for Pokémon
    start_search = time.time()
    pokemon_positions = []
    start_r, start_c = get_search_start(start)
    for i in range(start_r, start_r + SEARCH_DIM):
        for j in range(start_c, start_c + SEARCH_DIM):
            if is_pokemon(grid, (i, j)):
                pokemon_positions.append((i, j))
    end_search = time.time()
    search_time += (end_search - start_search)  # Update global timer

    max_score = 0
    best_path = path

    # Measure the time spent exploring Pokémon paths
    start_explore = time.time()
    for pokemon in pokemon_positions:
        dist = distance(start, pokemon)

        if depth - dist < 0:
            continue

        grid[pokemon[0]][pokemon[1]] = "."  # Remove from grid
        new_path = find_optimal_path_helper(grid, pokemon, path + [pokemon], depth - dist)
        grid[pokemon[0]][pokemon[1]] = "x"  # Restore to grid

        score = (depth - dist) + len(new_path)
        if score > max_score:
            max_score = score
            best_path = new_path
    end_explore = time.time()
    explore_time += (end_explore - start_explore)  # Update global timer

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
 
 
def find_optimal_path(grid, start, depth, distances):
    return find_optimal_path_helper(grid, start, [], depth, distances)
 

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


def calculate_distances(grid, start):
    distances = {}
    pokemon_positions = [start]

    for i in range(WIDTH):
        for j in range(HEIGHT):
            if is_pokemon(grid, (i, j)):
                pokemon_positions.append((i, j))

    for pokemon in pokemon_positions:
        for other in pokemon_positions:
            if pokemon == other and not is_pokemon(grid, start):
                continue
            dist = distance(pokemon, other)
            distances[(pokemon, other)] = dist
            
    return distances


def main():
    global WIDTH, HEIGHT, SEARCH_DIM

    start = (0,0)
    depth = int(sys.argv[2])
    grid, dimensions = read_data("data.txt")
    WIDTH, HEIGHT, workers = dimensions

    if SEARCH_DIM > min(WIDTH, HEIGHT):
        SEARCH_DIM = min(WIDTH, HEIGHT)
                
    print("[*] Calculating distances...")

    start_time = time.time()
    distances = calculate_distances(grid, start)
    end_time = time.time()
    dist_time = end_time - start_time
    
    print("[*] Finding optimal path...")

    start_time = time.time()
    path = find_optimal_path(grid, start, depth, distances)
    end_time = time.time()
    find_time = end_time - start_time
    
    moves = pokemons_to_moves(start, path)
    visualize_search(grid, moves, start)

    print(f"Time taken to calculate: {dist_time:.2f}s")
    print(f"Time taken to find path: {find_time:.2f}s")

    print(f"    Search time: {search:.2f}s")
    print(f"    Recursion time: {recursion:.2f}s")

    print(path)
    print(f"Len: {len(path)}")
    print(moves)
    print(f"moves: {len(moves)}")



if __name__ == '__main__':
    main()