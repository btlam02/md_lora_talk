"""Tiện ích landmark và các chỉ số MediaPipe FaceMesh dùng trong LoRA-Talk."""

from typing import FrozenSet, Set, Tuple, List


def summarize_landmarks(edge_set: FrozenSet[Tuple[int, int]]) -> Set[int]:
    """Lấy tập ID landmark duy nhất xuất hiện trong tập cạnh FaceMesh."""
    landmarks = set()
    for a, b in edge_set:
        landmarks.add(a)
        landmarks.add(b)
    return landmarks


FACEMESH_LIPS = frozenset(
    [
        (61, 146), (146, 91), (91, 181), (181, 84), (84, 17), (17, 314),
        (314, 405), (405, 321), (321, 375), (375, 291), (61, 185), (185, 40),
        (40, 39), (39, 37), (37, 0), (0, 267), (267, 269), (269, 270),
        (270, 409), (409, 291), (78, 95), (95, 88), (88, 178), (178, 87),
        (87, 14), (14, 317), (317, 402), (402, 318), (318, 324), (324, 308),
        (78, 191), (191, 80), (80, 81), (81, 82), (82, 13), (13, 312),
        (312, 311), (311, 310), (310, 415), (415, 308),
    ]
)

FACEMESH_LEFT_EYE = frozenset(
    [
        (263, 249), (249, 390), (390, 373), (373, 374), (374, 380), (380, 381),
        (381, 382), (382, 362), (263, 466), (466, 388), (388, 387), (387, 386),
        (386, 385), (385, 384), (384, 398), (398, 362),
    ]
)

FACEMESH_LEFT_EYEBROW = frozenset(
    [(276, 283), (283, 282), (282, 295), (295, 285), (300, 293), (293, 334), (334, 296), (296, 336)]
)

FACEMESH_RIGHT_EYE = frozenset(
    [
        (33, 7), (7, 163), (163, 144), (144, 145), (145, 153), (153, 154),
        (154, 155), (155, 133), (33, 246), (246, 161), (161, 160), (160, 159),
        (159, 158), (158, 157), (157, 173), (173, 133),
    ]
)

FACEMESH_RIGHT_EYEBROW = frozenset(
    [(46, 53), (53, 52), (52, 65), (65, 55), (70, 63), (63, 105), (105, 66), (66, 107)]
)

FACEMESH_FACE_OVAL = frozenset(
    [
        (389, 356), (356, 454), (454, 323), (323, 361), (361, 288), (288, 397),
        (397, 365), (365, 379), (379, 378), (378, 400), (400, 377), (377, 152),
        (152, 148), (148, 176), (176, 149), (149, 150), (150, 136), (136, 172),
        (172, 58), (58, 132), (132, 93), (93, 234), (234, 127), (127, 162),
    ]
)

FACEMESH_NOSE = frozenset(
    [(168, 6), (6, 197), (197, 195), (195, 5), (5, 4), (4, 45), (45, 220), (220, 115), (115, 48), (4, 275), (275, 440), (440, 344), (344, 278)]
)

FACEMESH_CONNECTION = frozenset().union(
    FACEMESH_LIPS,
    FACEMESH_LEFT_EYE,
    FACEMESH_LEFT_EYEBROW,
    FACEMESH_RIGHT_EYE,
    FACEMESH_RIGHT_EYEBROW,
    FACEMESH_FACE_OVAL,
    FACEMESH_NOSE,
)

ALL_LANDMARK_IDX = summarize_landmarks(FACEMESH_CONNECTION)
POSE_LANDMARK_IDX = summarize_landmarks(
    FACEMESH_NOSE.union(FACEMESH_RIGHT_EYEBROW, FACEMESH_RIGHT_EYE, FACEMESH_LEFT_EYE, FACEMESH_LEFT_EYEBROW)
).union([162, 127, 234, 93, 389, 356, 454, 323])
CONTENT_LANDMARK_IDX = ALL_LANDMARK_IDX - POSE_LANDMARK_IDX

ORI_SEQUENCE_IDX = [
    162, 127, 234, 93, 132, 58, 172, 136, 150, 149, 176, 148, 152, 377, 400, 378,
    379, 365, 397, 288, 361, 323, 454, 356, 389, 70, 63, 105, 66, 107, 55, 65,
    52, 53, 46, 336, 296, 334, 293, 300, 276, 283, 282, 295, 285, 168, 6, 197,
    195, 5, 48, 115, 220, 45, 4, 275, 440, 344, 278, 33, 246, 161, 160, 159,
    158, 157, 173, 133, 155, 154, 153, 145, 144, 163, 7, 362, 398, 384, 385,
    386, 387, 388, 466, 263, 249, 390, 373, 374, 380, 381, 382, 61, 185, 40, 39,
    37, 0, 267, 269, 270, 409, 291, 375, 321, 405, 314, 17, 84, 181, 91, 146, 78,
    191, 80, 81, 82, 13, 312, 311, 310, 415, 308, 324, 318, 402, 317, 14, 87,
    178, 88, 95,
]

ORDER = {lm_id: i for i, lm_id in enumerate(ORI_SEQUENCE_IDX)}
POSE_SEQUENCE_IDX = sorted(list(POSE_LANDMARK_IDX), key=lambda x: ORDER[x])
CONTENT_SEQUENCE_IDX = sorted(list(CONTENT_LANDMARK_IDX), key=lambda x: ORDER[x])

IMG_SIZE = 128
MOUTH_SLICE = (slice(72, 116), slice(34, 94))
MOUTH_PRIOR_SLICE = (slice(66, 122), slice(24, 104))
LOWER_SLICE = (slice(IMG_SIZE // 2, IMG_SIZE), slice(0, IMG_SIZE))
