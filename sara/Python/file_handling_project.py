from pathlib import Path
import os

def readFileAndFolder():
    path = Path("")
    items = list(path.rglob("*"))
    for i, items in enumerate(items, 1):
        print(f"{i+1}. {items}")

def createFile():
    try:
        readFileAndFolder()
        file_name = input("Enter the name of the file: ")
        p = Path(file_name)
        if not p.exists():
            with open(p, "w") as fs:
                data = input("What you want to write in the file: ")
                fs.write(data)
            print(f"File {file_name} created successfully")
        else:
            print(f"File {file_name} already exists")

    except Exception as err:
        print(f"There is an error: {err}")

def readFile():
    try:
        readFileAndFolder()
        file_name = input("Which file you want to read: ")
        p = Path(file_name)
        if p.exists() and p.is_file():
            with open(p, "r") as fs:
                data = fs.read()
                print(data)
            print(f"File {file_name} read successfully")
        else:
            print(f"File {file_name} does not exist")
    except Exception as err:
        print(f"There is an error: {err}")

def updateFile():
    try:
        readFileAndFolder()
        file_name = input("Which file you want to update: ")
        p = Path(file_name)
        if p.exists() and p.is_file():
            print("Press 1 for changing the file name")
            print("Press 2 for overwriting the file content")
            print("Press 3 for appending some content in the file")

            response = int(input("Tell your response: "))

            if response == 1:
                name2 = input("Enter the new name of the file: ")
                p2 = Path(name2)
                p.rename(p2)
                print(f"File {p} renamed to {p2}")

            if response == 2:
                with open(p, "w") as fs:
                    data = input("Enter the new content of the file to overwrite the old content: ")
                    fs.write(data)
                    print(f"File {p} overwritten successfully")

            if response == 3:
                with open(p, "a") as fs:
                    data = input("Enter the content you want to append in the file: ")
                    fs.write(" " + data)
                print(f"Content appended to {p} successfully")

    except Exception as err:
        print(f"There is an error: {err}")

def deleteFile():
    try:
        readFileAndFolder()
        name = input("Which file you want to delete: ")
        p = Path(name)
        if p.exists() and p.is_file():
            os.remove(p)
            print(f"File {p} deleted successfully")
        else:
            print(f"File {p} does not exist")
    except Exception as err:
        print(f"There is an error: {err}")

print("Press 1 for creating a file")
print("Press 2 for reading a file")
print("Press 3 for updating a file")
print("Press 4 for deleting a file")

check = int(input("Tell your response: "))

if check == 1:
    createFile()
elif check == 2:
    readFile()
elif check == 3:
    updateFile()
elif check == 4:
    deleteFile()