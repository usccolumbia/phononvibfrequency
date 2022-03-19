import os

def getHighestNumAtoms(in_dir, out_dir):
    if (in_dir[-1] != '/'):
        in_dir += '/'
    if (out_dir[-1] != '/'):
        out_dir += '/'    
    directory = os.listdir(in_dir)

    maxCount = 0
    for f in directory:
        try:
            poscar = open(in_dir + f + '/' + "POSCAR", 'r')
            poscar_lines = poscar.readlines()
            atom_counts = poscar_lines[6].rstrip("\n").split()
            no_of_atoms = 0
            for i in atom_counts:
                no_of_atoms += eval(i)
            if (no_of_atoms > maxCount):
                maxCount = no_of_atoms
        except Exception as e:
            error = open(out_dir +"errors.txt", 'a')
            error.write(f'Error with parsing atom counts, id {f}: ' + repr(e) + '\n')
            error.close()
    return maxCount

def extract_frequencies(in_dir, out_dir):
    print(">>> Finding the highest number of atoms in the data set...")
    n = getHighestNumAtoms(in_dir, out_dir)
    if (in_dir[-1] != '/'):
        in_dir += '/'
    if (out_dir[-1] != '/'):
        out_dir += '/'

    errors = open(out_dir +"errors.txt", 'a')
    writeline = ""
    DELIM = ','
    directory = os.listdir(in_dir)

    print(">>> Extracting vibration frequencies...")
    with open(out_dir + "vibrationfrequency.csv", 'w') as outfile:
        outfile.write("ID,Formula,Num_Atoms")
        for p in range(3*n):
            outfile.write(f',P_{p}')
        outfile.write('\n')
        for structure_id in directory:
            try:
                writeline += structure_id + DELIM
                poscar = open(in_dir + structure_id + '/' + "POSCAR", 'r')
                poscar_lines = poscar.readlines()
                elements = poscar_lines[5].rstrip("\n").split()
                formula_numbers = poscar_lines[6].rstrip("\n").split() # analogous to atom counts

                formula = ""
                for i in range(len(elements)):
                    formula += elements[i] + formula_numbers[i]
                
                no_of_atoms = 0
                for i in formula_numbers:
                    no_of_atoms += eval(i)

                writeline += formula + DELIM + str(no_of_atoms)
            
                poscar.close()

                outcar = open(in_dir + structure_id + '/' + "OUTCAR", 'r')
                outcar_lines = [line.rstrip('\n').split() for line in outcar.readlines()]
                outcar_lines = [x for x in outcar_lines if len(x) > 0]

                which_frequency = 1 #starts out with 1st frequency
                
                
                for i in outcar_lines:
                    if (i[0] == str(which_frequency)):
                        if (i[1] == "f"):
                            writeline += DELIM + i[3]
                        else:
                            writeline += DELIM + "-" + i[2]
                        which_frequency += 1
                
                for i in range(3*n-3*no_of_atoms):
                    writeline += DELIM + "0"
                
                outcar.close()

                writeline += "\n"
                outfile.write(writeline)
                writeline = ""
            except Exception as e:
                error = open(out_dir +"errors.txt", 'a')
                error.write(f'Error with parsing poscars and outcars, id {structure_id}: ' + repr(e) + '\n')
                error.close()
