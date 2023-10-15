def mapValueChar(val):
    if val < 50:
        return "."
    if val < 150:
        return "*"
    return "@"

def printImage(image):
    for l in image:
        print("".join(map(mapValueChar, l)))