using PyCall


hoomd = pyimport("hoomd")
md = pyimport("hoomd.md")
np = pyimport("numpy")


hoomd.context.initialize("--mode=gpu")
hoomd.option.set_num_threads(8)


const kT = 0.596161
const dt = 0.02045


# System Definition
# -----------------

## Set snapshot
snapshot = hoomd.data.make_snapshot(
    N = 14,
    box = hoomd.data.boxdim(Lx = 41, Ly = 41, Lz = 41),
    particle_types = ["C", "H"],
    bond_types = ["CC", "CH"],
    angle_types = ["CCC", "CCH", "HCH"],
    dihedral_types = ["CCCC", "HCCC", "HCCH"],
    pair_types = ["CCCC", "HCCC", "HCCH"],
    dtype = "double"
)

## Set particles types ids
let ids = PyArray(snapshot.particles."typeid")
    ids[[2:4; 6:7; 9:10; 12:14]] .= 1
end

## Compute initial positions
positions = [
    -2.990196   0.097881   0.000091;
    -2.634894  -0.911406   0.001002;
    -2.632173   0.601251  -0.873601;
    -4.060195   0.099327  -0.000736;
    -2.476854   0.823942   1.257436;
    -2.832157   1.833228   1.256526;
    -2.834877   0.320572   2.131128;
    -0.936856   0.821861   1.258628;
    -0.578833   1.325231   0.384935;
    -0.581553  -0.187426   1.259538;
    -0.423514   1.547922   2.515972;
    -0.781537   1.044552   3.389664;
    +0.646485   1.546476   2.516800;
    -0.778816   2.557208   2.515062;
]

reference_box_low_coords = [-22.206855  -19.677099  -19.241968]
box_low_coords = [-snapshot.box.Lx/2  -snapshot.box.Ly/2  -snapshot.box.Lz/2]
positions .+= box_low_coords .- reference_box_low_coords

PyArray(snapshot.particles."position") .= positions

## Set masses
let
    mC = 12.00
    mH = 1.008
    PyArray(snapshot.particles."mass") .= [
        mC, mH, mH, mH,
        mC, mH, mH,
        mC, mH, mH,
        mC, mH, mH, mH
    ]
end

## Set charges
let
    reference_charges = [
        -0.180000, 0.060000, 0.060000, 0.060000,
        -0.120000, 0.060000, 0.060000,
        -0.120000, 0.060000, 0.060000,
        -0.180000, 0.060000, 0.060000, 0.060000
    ]
    charge_conversion = 18.22262
    PyArray(snapshot.particles."charge") .= charge_conversion .* reference_charges
end

## Set bonds
snapshot.bonds.resize(13)

let bonds_ids = PyArray(snapshot.bonds."typeid")
    bonds_ids[[1:3; 5:6; 8:9; 11:13]] .= 1
end

PyArray(snapshot.bonds."group") .= [
    0   2;
    0   1;
    0   3;
    0   4;
    4   5;
    4   6;
    4   7;
    7   8;
    7   9;
    7   10;
    10  11;
    10  12;
    10  13;
]

## Set angles
snapshot.angles.resize(24)

let angles_ids = PyArray(snapshot.angles."typeid")
    angles_ids[[3; 5:8; 11:14; 17:21]] .= 1
    angles_ids[[1:2; 4; 10; 16; 22:24]] .= 2
end

PyArray(snapshot.angles."group") .= [
    1   0   2 ;
    2   0   3 ;
    2   0   4 ;
    1   0   3 ;
    1   0   4 ;
    3   0   4 ;
    0   4   5 ;
    0   4   6 ;
    0   4   7 ;
    5   4   6 ;
    5   4   7 ;
    6   4   7 ;
    4   7   8 ;
    4   7   9 ;
    4   7   10;
    8   7   9 ;
    8   7   10;
    9   7   10;
    7   10  11;
    7   10  12;
    7   10  13;
    11  10  12;
    11  10  13;
    12  10  13;
]

## Set dihedrals 
snapshot.dihedrals.resize(27)

let dihedrals_ids = PyArray(snapshot.dihedrals."typeid")
    dihedrals_ids[[3; 6; 9:11; 15; 18:21]] .= 1
    dihedrals_ids[[1:2; 4:5; 7:8; 13:14; 16:17; 22:27]] .= 2
end

PyArray(snapshot.dihedrals."group") .= [
    2   0   4   5 ;
    2   0   4   6 ;
    2   0   4   7 ;
    1   0   4   5 ;
    1   0   4   6 ;
    1   0   4   7 ;
    3   0   4   5 ;
    3   0   4   6 ;
    3   0   4   7 ;
    0   4   7   8 ;
    0   4   7   9 ;
    0   4   7   10;
    5   4   7   8 ;
    5   4   7   9 ;
    5   4   7   10;
    6   4   7   8 ;
    6   4   7   9 ;
    6   4   7   10;
    4   7   10  11;
    4   7   10  12;
    4   7   10  13;
    8   7   10  11;
    8   7   10  12;
    8   7   10  13;
    9   7   10  11;
    9   7   10  12;
    9   7   10  13;
]

## Set pairs
snapshot.pairs.resize(27)

let pairs_ids = PyArray(snapshot.pairs."typeid")
    pairs_ids[2:11] .= 1
    pairs_ids[12:27] .= 2
end

PyArray(snapshot.pairs."group") .= [
    # CCCC
    0   10;
    # HCCC\n",
    0   8 ;
    0   9 ;
    5   10;
    6   10;
    1   7 ;
    2   7 ;
    3   7 ;
    11  4 ;
    12  4 ;
    13  4 ;
    # HCCH
    1   5 ;
    1   6 ;
    2   5 ;
    2   6 ;
    3   5 ;
    3   6 ;
    5   8 ;
    6   8 ;
    5   9 ;
    6   9 ;
    8   11;
    8   12;
    8   13;
    9   11;
    9   12;
    9   13;
]

## Materialize system from snapshot
system = hoomd.init.read_snapshot(snapshot)


# Interactions

## Neighbor list
nl_ex = md.nlist.cell()
nl_ex.reset_exclusions(exclusions = ["1-2", "1-3", "1-4"])

## Lennard-Jones interactions
lj = md.pair.lj(r_cut = 12.0, nlist = nl_ex)
lj.pair_coeff.set("C", "C", epsilon = 0.07, sigma = 3.55)
lj.pair_coeff.set("H", "H", epsilon = 0.03, sigma = 2.42)
lj.pair_coeff.set("C", "H", epsilon = sqrt(0.07 * 0.03), sigma = sqrt(3.55 * 2.42))

## Coulomb interactions
coulomb = md.charge.pppm(hoomd.group.charged(), nlist = nl_ex)
coulomb.set_params(Nx = 64, Ny = 64, Nz = 64, order = 6, rcut = 12.0)

## Harmonic interactions
harmonic = md.bond.harmonic()
harmonic.bond_coeff.set("CC", k = 2 * 268.0, r0 = 1.529)
harmonic.bond_coeff.set("CH", k = 2 * 340.0, r0 = 1.09)

## Angle potentials
angle = md.angle.harmonic()
angle.angle_coeff.set("CCC", k = 2 * 58.35, t0 = 112.7 * π / 180)
angle.angle_coeff.set("CCH", k = 2 * 37.5, t0 = 110.7 * π / 180)
angle.angle_coeff.set("HCH", k = 2 * 33.0, t0 = 107.8 * π / 180)

## Dihedral potentials
dihedral = md.dihedral.opls()
dihedral.dihedral_coeff.set("CCCC", k1 = 1.3, k2 = -0.05, k3 = 0.2, k4 = 0.0)
dihedral.dihedral_coeff.set("HCCC", k1 = 0.0, k2 = 0.0, k3 = 0.3, k4 = 0.0)
dihedral.dihedral_coeff.set("HCCH", k1 = 0.0, k2 = 0.0, k3 = 0.3, k4 = 0.0)

## Special Lennard-Jones interactions
lj_special_pairs = md.special_pair.lj()
lj_special_pairs.pair_coeff.set("CCCC", epsilon = 0.07, sigma = 3.55, r_cut = 12.0)
lj_special_pairs.pair_coeff.set("HCCH", epsilon = 0.03, sigma = 2.42, r_cut = 12.0)
lj_special_pairs.pair_coeff.set("HCCC",
    epsilon = sqrt(0.07 * 0.03), sigma = sqrt(3.55 * 2.42), r_cut = 12.0
)

## Special Coulomb interactions
coulomb_special_pairs = md.special_pair.coulomb()
coulomb_special_pairs.pair_coeff.set("CCCC", alpha = 0.5, r_cut = 12.0)
coulomb_special_pairs.pair_coeff.set("HCCC", alpha = 0.5, r_cut = 12.0)
coulomb_special_pairs.pair_coeff.set("HCCH", alpha = 0.5, r_cut = 12.0)


# Integration method
# ------------------

md.integrate.mode_standard(dt = dt)
integrator = md.integrate.nvt(group = hoomd.group.all(), kT = kT, tau = 100dt)
integrator.randomize_velocities(seed = 42)
