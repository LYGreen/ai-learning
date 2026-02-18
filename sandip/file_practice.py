with open("test.txt", "w") as f:
    f.write("This is my first file\n")
    f.write("Learning Python file I/O")

with open("test.txt", "r") as f:
    content = f.read()
    print(content)
