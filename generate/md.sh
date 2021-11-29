# ./xtb/build/xtb 1-cyclohexylmethyl-4-methoxy-benzene.xyz --input scan.inp --md
../../xtb/xtb/build/xtb "base_structs/struct_$1.xyz" --input scan.inp --md
mv xtb.trj xtb.trj.xyz
mkdir "../data/xtb_data/base_$1"
mkdir "../data/xtb_data/base_$1/STRUCTS/"
python3 split_xyz_files.py 2 xtb.trj.xyz "../data/xtb_data/base_$1"
