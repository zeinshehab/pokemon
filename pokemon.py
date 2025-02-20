from collections import deque
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


# def get_subgrid(grid, pos):
# 	search_pos = (int(pos[0]-SEARCH_DIM/2), int(pos[1]-SEARCH_DIM/2))

# 	search_grid = grid[search_pos[0]:(search_pos[0]+SEARCH_DIM)]
# 	search_grid = [x[search_pos[1]:(search_pos[1]+SEARCH_DIM)] for x in search_grid]

# 	return search_grid

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


def best_path_helper(grid, start, positions, moves, depth):
	print(f"Depth: {depth} | Moves: {moves}\n")


	if moves <= 0 or depth <= 0:
		return len(positions), positions

	pos = start

	pokemon_positions = []

	start_r, start_c = get_search_start(pos)
	for i in range(start_r, start_r + SEARCH_DIM):
	    for j in range(start_c, start_c + SEARCH_DIM):
	        if is_pokemon(grid, (i, j)):
	        	pokemon_positions.append((i, j))

	max_score = 0
	best_positions = positions

	for dest in pokemon_positions:
		dist = distance(pos, dest)

		new_positions = positions + [dest]
		updated_grid = [row[::] for row in grid]
		updated_grid[dest[0]][dest[1]] = '.'
		# grid[dest[0]][dest[1]] = "."

		score, positions = best_path_helper(updated_grid, dest, new_positions, moves-dist, depth-1)
		score += REWARD

		# grid[dest[0]][dest[1]] = 'x'

		if score > max_score:
			max_score = score
			best_positions = positions

	return max_score, best_positions




def best_path(grid, start, depth):
	return best_path_helper(grid, start, [], MAX_PATH_LEN, depth) 


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


def main():
	global WIDTH, HEIGHT

	starts = [(0, 0)]
	grid, dimensions = read_data("data-big.txt")
	WIDTH, HEIGHT, workers = dimensions

	for worker in range(workers):
		start = starts[worker]
		# start = (int(WIDTH/2), int(HEIGHT/2))

		score, best = best_path(grid, start, 20)
		# visualize_search(grid, best_path, start)

		# set_at(grid, start, "O")
		# sub = get_subgrid(grid, start)
		# print(sub)

		# print_grid(sub)

		# print(len(sub))
		# print(len(sub[0]))

		print(best)
		print(f"Len: {len(best)}")
		print(score)


if __name__ == '__main__':
	main()