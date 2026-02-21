# Object Oriented Programming

class Factory:
    def __init__(self, material, zips, pockets):
        self.material = material
        self.zips = zips
        self.pockets = pockets

    def show(self):
        print(f"This is a {self.material} bag with {self.zips} zips and {self.pockets} pockets")


reebok = Factory("leather", 10, 8)
campus = Factory("nylon", 8, 5)
reebok.show()
campus.show()
print(reebok.__dict__)
print(campus.__dict__)