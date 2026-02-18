# File Handling

# Mode: r, w, a, x
# Description:
# r = Read (default) - file must exist
# w = Write - create file or overwrite
# a = Append - add to end of file
# s = Create - create a new file, fails if exists 

# Writing to a file
r = open("file.txt", "w")

r.write("Now I am writing to the file")

r.close()