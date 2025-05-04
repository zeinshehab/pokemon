# Pokemon

## Description
- This program finds the best start position along with the best path and set of moves starting from that position to capture the most pokemons.
- Output is a single start position along with a set of moves to make (the path) in order to capture the most pokemons.

>[!NOTE]
> I have added a file `results.txt` that contains optimal paths and starting positions that I generated on HIGH accuracy in interest of saving time in case you want to test.

## Usage

### Brief description of command line arguments:
- `search_dim_x and search_dim_y`: The rectangle area where you search around you for pokemon. Usually small for dense matrices and larger for sparse ones. Bigger is slower.
- `max_num_moves`: How many moves you are allowed. Typically 100 in our case.
- `data_file`: Path of data file to load. Format: `data/<file_name>`
- `prune_start_positions`: `1` or `0`. Decides whether to treat each start position as an independent search or if allowed to take advantage of earlier searches and prune bad positions. Much faster at `1`.
- `acuuracy`: `0`, `1`, `2`, `3`. LOW, MEDIUM, HIGH, ROUND respectively.
    - LOW: will run the fastest, is not guaranteed the best solution.
    - MEDIUM: Slower but still fast, will give a good enough solution close to the theoretical average. Should be the most reasonable choice even for large file unless you require the top optimal.
    - HIGH: Slowest. will give you best solution. 
    - ROUND: mostly used for debug purposes. will perform similar to high mostly

>[!WARNING]
> - In all of these cases, algorithm will run much faster if you find a good start position early so it might hang a bit (a few seconds on an initial start position and then speed through the rest). Speed also depends on the size of the file and sparsity.
> - All files up until file `05_corridor.txt` (inclusive) will give the optimal solution on HIGH accuracy in under 20 minutes (on medium in seconds except for corridor takes around 15min).
> - Keep in mind that tuning these specific parameters for each individual file based on how its organized could yield better results but in general you should get acceptable solutions.

### Command
- command `python main.py <search_dim_x> <search_dim_y> <max_num_moves> <data_file> <prune_start_positions> <accuracy>`

### Example Usage
For small file:  `python main.py 5 5 100 data/01_simple_example.txt 1 2`
For medium file: `python main.py 5 5 100 data/03_medium.txt 1 1`
For big file:    `python main.py 5 5 100 data/06_big.txt 1 1`