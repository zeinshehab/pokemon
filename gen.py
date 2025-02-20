import random

data_file = open("data-sparse.txt", "w+")

width, height, workers = 200, 100, 1

data = f"{width} {height} {workers}\n"
for i in range(height):
	line = ""
	for j in range(width):
		choice = random.random()
		if choice <= 0.1:
			line += "x"
		else:
			line += "."
	data += line + "\n"

data_file.write(data)