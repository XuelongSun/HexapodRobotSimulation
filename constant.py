# useful
AXIS_INDEX = {'X':0, 'Y':1, 'Z':2}

# hexapod robot
DEFAULT_DIMSIONS = (2, 4, 4)

# legs
DEFAULT_LEG_LENGTH = (2, 2, 2)
DEFAULT_LEG_ALPHA_BIAS = (-90, -45, 45, 90, 135, -135)
DEFAULT_LEG_GAMMA = -90
LEG_ID_NAMES = {0: "MiddleRight", 1:"FrontRight", 2:"FrontLeft",
                3: "MiddleLeft", 4:"RearLeft", 5:"RearRight"}
LEG_NAMES_ID = {}
for k,v in LEG_ID_NAMES.items():
    LEG_NAMES_ID[v] = k
LEG_SEG_ID_NAMES = {0:"coxa", 1:"femur", 2:"tibia"}
LEG_SEG_NAMES_ID = {}
for k, v in LEG_SEG_ID_NAMES.items():
    LEG_SEG_NAMES_ID[v] = k

GOOD_LEG_TRIOS = [
    (0, 1, 3),
    (0, 1, 4),
    (0, 2, 3),
    (0, 2, 4),
    (0, 2, 5),
    (0, 3, 4),
    (0, 3, 5),
    (1, 2, 4),
    (1, 2, 5),
    (1, 3, 4),
    (1, 3, 5),
    (1, 4, 5),
    (2, 3, 5),
    (2, 4, 5),
]

ADJACENT_LEG_TRIOS = [
    (0, 1, 2),
    (1, 2, 3),
    (2, 3, 4),
    (3, 4, 5),
    (0, 4, 5),
    (0, 1, 5),
]

LEG_TRIOS = GOOD_LEG_TRIOS + ADJACENT_LEG_TRIOS
