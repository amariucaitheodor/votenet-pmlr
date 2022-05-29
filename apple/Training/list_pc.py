from os import listdir
from os.path import isfile, join
onlyfiles = [f.split("_")[0] for f in listdir("./data") if isfile(join("./data", f))]

print(onlyfiles)
