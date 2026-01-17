import sys
sys.path.append('.')
import tracea
with open('tracea_dir.txt', 'w') as f:
    f.write(str(dir(tracea)))
    f.write('\n')
    if hasattr(tracea, 'PyEpilogueOp'):
        f.write('PyEpilogueOp members: ' + str(dir(tracea.PyEpilogueOp)) + '\n')
