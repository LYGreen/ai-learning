class fac():
    def __init__(self, material, zips):
        self.material = material
        self.zips = zips

class fac2(fac):
    def __init__(self, material, zips, pockets):
        super().__init__(material, zips)
        self.pockets = pockets


class fac3(fac2):
    def __init__(self, material, zips, pockets, straps):
        super().__init__(material, zips, pockets)
        self.straps = straps

obj = fac3("leather", 10, 8, 2)