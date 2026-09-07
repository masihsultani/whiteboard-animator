import numpy as np

from whiteboard_animator import animator as animator_module


class _NoopCraftDetector:
    def __init__(self, *args, **kwargs):
        pass

    def detect(self, image):
        return np.zeros(image.shape[:2], dtype=bool)


animator_module.CRAFTDetector = _NoopCraftDetector

from whiteboard_animator.animator import (  # noqa: E402
    WhiteboardAnimator,
    assign_components_to_boxes,
)


def make_component(left, top, right, bottom, *, is_fill=False, is_text=False):
    mask = np.zeros((1000, 1000), dtype=bool)
    mask[top:bottom, left:right] = True
    return {
        "left": left,
        "top": top,
        "right": right,
        "bottom": bottom,
        "area": (right - left) * (bottom - top),
        "is_fill": is_fill,
        "is_text": is_text,
        # Real components carry numpy arrays, which make == comparisons
        # (list.remove/index) blow up; the fixture must too.
        "mask": mask,
        "solid_mask": mask.copy(),
    }


def make_animator() -> WhiteboardAnimator:
    return WhiteboardAnimator()


# ── assign_components_to_boxes ──────────────────────────────────────────


def test_containment_prefers_smallest_box():
    big = [0, 0, 1000, 1000]
    small = [100, 100, 300, 300]
    comp = make_component(150, 150, 250, 250)

    per_box, unmatched = assign_components_to_boxes([comp], [big, small])

    assert per_box == [[], [comp]]
    assert unmatched == []


def test_overlap_pass_matches_partially_covered_component():
    box = [0, 0, 200, 200]
    # Center (250, 100) is outside, but half the component overlaps the box.
    comp = make_component(100, 50, 400, 150)

    per_box, unmatched = assign_components_to_boxes([comp], [box])

    assert per_box == [[comp]]
    assert unmatched == []


def test_proximity_pass_attaches_nearby_component():
    box = [0, 0, 400, 400]
    # Outside, no overlap, but close to the box center.
    comp = make_component(410, 180, 440, 220)

    per_box, unmatched = assign_components_to_boxes([comp], [box])

    assert per_box == [[comp]]
    assert unmatched == []


def test_far_component_is_unmatched():
    box = [0, 0, 100, 100]
    comp = make_component(900, 900, 950, 950)

    per_box, unmatched = assign_components_to_boxes([comp], [box])

    assert per_box == [[]]
    assert unmatched == [comp]


def test_none_box_gets_no_components():
    box = [0, 0, 1000, 1000]
    comp = make_component(100, 100, 200, 200)

    per_box, unmatched = assign_components_to_boxes([comp], [None, box])

    assert per_box == [[], [comp]]
    assert unmatched == []


# ── _schedule_components (legacy behavior) ──────────────────────────────


def test_schedule_components_is_sequential_and_area_weighted():
    animator = make_animator()
    comp_a = make_component(0, 0, 100, 100)      # area 10000, sqrt 100
    comp_b = make_component(0, 200, 300, 500)    # area 90000, sqrt 300

    schedule = animator._schedule_components([comp_a, comp_b], 4.0)

    (first, start_a, dur_a), (second, start_b, dur_b) = schedule
    assert first is comp_a and second is comp_b
    assert start_a == 0.0
    assert abs(dur_a - 1.0) < 1e-6      # 4.0 * 100/400
    assert abs(start_b - dur_a) < 1e-6
    assert abs(dur_b - 3.0) < 1e-6      # 4.0 * 300/400


def test_schedule_components_caps_fill_duration():
    animator = make_animator()
    fill = make_component(0, 0, 1000, 1000, is_fill=True)

    schedule = animator._schedule_components([fill], 20.0)

    _, _, dur = schedule[0]
    assert dur == animator.max_fill_duration


# ── _schedule_elements ──────────────────────────────────────────────────


def element_entry(boxes, start, end):
    return {"boxes": boxes, "start": start, "end": end}


def norm_box(ymin, xmin, ymax, xmax):
    return {"ymin": ymin, "xmin": xmin, "ymax": ymax, "xmax": xmax}


def test_groups_use_natural_durations_then_hold():
    animator = make_animator()
    shape = (1000, 1000)
    top_comp = make_component(100, 100, 300, 200)
    bottom_comp = make_component(100, 600, 300, 800)
    plan = [
        element_entry([norm_box(0, 0, 400, 1000)], 0.0, 4.0),
        element_entry([norm_box(500, 0, 1000, 1000)], 4.0, 9.0),
    ]

    schedule = animator._schedule_elements([top_comp, bottom_comp], plan, shape)

    by_comp = {id(comp): (start, dur) for comp, start, dur in schedule}
    start_top, dur_top = by_comp[id(top_comp)]
    start_bottom, dur_bottom = by_comp[id(bottom_comp)]
    # Each semantic object starts with its cue but keeps a natural draw speed
    # instead of stretching a few pixels across the entire narration window.
    assert start_top == 0.0
    assert start_top + dur_top < 4.0
    assert start_bottom == 4.0
    assert start_bottom + dur_bottom < 9.0


def test_unmatched_components_draw_in_the_last_window():
    animator = make_animator()
    shape = (1000, 1000)
    matched = make_component(450, 450, 550, 550)
    stray = make_component(900, 0, 950, 50)  # far from both boxes
    plan = [
        element_entry([norm_box(0, 0, 100, 100)], 0.0, 3.0),
        element_entry([norm_box(400, 400, 600, 600)], 3.0, 6.0),
    ]

    schedule = animator._schedule_elements([matched, stray], plan, shape)

    starts = {id(comp): start for comp, start, _dur in schedule}
    assert starts[id(matched)] == 3.0
    assert starts[id(stray)] > starts[id(matched)]


def test_spanning_component_assigned_to_earliest_element():
    animator = make_animator()
    shape = (1000, 1000)
    # A chart frame spanning all three row bands; its center sits in the
    # middle band, which used to misassign it there.
    frame = make_component(40, 60, 960, 910)
    row_text = make_component(100, 400, 500, 550)
    plan = [
        element_entry([norm_box(62, 47, 343, 951)], 0.0, 3.0),
        element_entry([norm_box(348, 47, 601, 951)], 3.0, 6.0),
        element_entry([norm_box(606, 47, 907, 951)], 6.0, 9.0),
    ]

    schedule = animator._schedule_elements([frame, row_text], plan, shape)

    by_comp = {id(comp): start for comp, start, _dur in schedule}
    assert by_comp[id(frame)] == 0.0                      # earliest spanned element
    assert by_comp[id(row_text)] > by_comp[id(frame)]     # row content after the frame


def test_component_covering_single_box_is_not_treated_as_spanning():
    animator = make_animator()
    shape = (1000, 1000)
    # Fills its own band but only grazes the next one.
    big_in_band = make_component(60, 70, 940, 330)
    plan = [
        element_entry([norm_box(62, 47, 343, 951)], 0.0, 3.0),
        element_entry([norm_box(348, 47, 601, 951)], 3.0, 6.0),
    ]

    schedule = animator._schedule_elements([big_in_band], plan, shape)

    assert len(schedule) == 1
    assert schedule[0][1] == 0.0


def test_multi_box_element_collects_components_from_all_its_item_boxes():
    animator = make_animator()
    shape = (1000, 1000)
    # "Three curves side by side" — one element, three item boxes.
    left = make_component(60, 620, 280, 880)
    middle = make_component(390, 620, 610, 880)
    right = make_component(720, 620, 940, 880)
    title = make_component(300, 80, 700, 200)
    plan = [
        element_entry([norm_box(50, 250, 250, 750)], 0.0, 3.0),
        element_entry([
            norm_box(600, 40, 900, 300),
            norm_box(600, 370, 900, 630),
            norm_box(600, 700, 900, 960),
        ], 3.0, 8.0),
    ]

    schedule = animator._schedule_elements([left, middle, right, title], plan, shape)

    by_comp = {id(comp): start for comp, start, _dur in schedule}
    assert by_comp[id(title)] == 0.0
    # All three items draw after the first element's content, in order.
    for comp in (left, middle, right):
        assert by_comp[id(comp)] > by_comp[id(title)]
    assert by_comp[id(left)] < by_comp[id(middle)] < by_comp[id(right)]


def test_item_boxes_draw_in_detected_order_not_geometric_order():
    animator = make_animator()
    shape = (1000, 1000)
    # The detector says the LOWER item draws first (e.g. the main object),
    # then the upper one (e.g. its callout). Geometric sorting would flip it.
    lower = make_component(120, 620, 350, 880)
    upper = make_component(120, 70, 350, 330)
    plan = [
        element_entry([
            norm_box(600, 100, 900, 370),   # first in draw order
            norm_box(50, 100, 350, 370),    # second
        ], 0.0, 6.0),
    ]

    schedule = animator._schedule_elements([upper, lower], plan, shape)

    starts = {id(comp): start for comp, start, _dur in schedule}
    assert starts[id(lower)] < starts[id(upper)]


def test_text_draws_after_shape_within_an_item():
    animator = make_animator()
    shape = (1000, 1000)
    # Label sits ABOVE the shape inside the same item box; the hand still
    # draws the shape first, then names it.
    label = make_component(140, 110, 320, 170)
    label["is_text"] = True
    shape_comp = make_component(100, 200, 360, 460)
    plan = [
        element_entry([norm_box(80, 80, 500, 400)], 0.0, 5.0),
    ]

    schedule = animator._schedule_elements([label, shape_comp], plan, shape)

    starts = {id(comp): start for comp, start, _dur in schedule}
    assert starts[id(shape_comp)] < starts[id(label)]


def test_overlay_draws_after_complete_text_object():
    animator = make_animator()
    shape = (1000, 1000)
    word = make_component(100, 300, 500, 420, is_text=True)
    cross = make_component(180, 250, 420, 470)
    cross["_is_overlay"] = True
    plan = [
        element_entry([norm_box(200, 50, 500, 550)], 0.0, 4.0),
    ]

    schedule = animator._schedule_elements([cross, word], plan, shape)

    starts = {id(comp): start for comp, start, _dur in schedule}
    assert starts[id(word)] < starts[id(cross)]


def test_many_children_cannot_overflow_semantic_parent_window():
    animator = make_animator()
    children = [
        make_component(index * 3, 100, index * 3 + 2, 110, is_text=True)
        for index in range(200)
    ]

    schedule = animator._schedule_components_fitted(
        children, duration=1.4, start=3.0,
    )

    assert len(schedule) == len(children)
    assert abs(max(start + duration for _comp, start, duration in schedule) - 4.4) < 1e-6


def test_semantic_parent_mask_is_exact_union_of_disconnected_children():
    animator = make_animator()
    left = make_component(50, 50, 100, 100)
    right = make_component(300, 50, 350, 100)

    obj = animator._make_semantic_object([left, right])

    import numpy as np
    expected = left["mask"] | right["mask"]
    assert np.array_equal(obj["mask"], expected)
    assert obj["children"][0] is left
    assert obj["children"][1] is right


def test_text_color_split_preserves_pixels_and_marks_overlay():
    import numpy as np

    animator = make_animator()
    image = np.full((80, 180, 3), 255, dtype=np.uint8)
    image[30:50, 20:150] = (20, 20, 20)
    image[20:60, 78:86] = (220, 25, 25)
    mask = np.any(image < 240, axis=2)
    comp = animator._build_component(
        mask, int(mask.sum()), is_text_override=True,
    )

    split = animator._split_text_color_overlays([comp], image)

    overlays = [item for item in split if item.get("_is_overlay")]
    base = [item for item in split if not item.get("_is_overlay")]
    assert overlays and base
    reconstructed = np.zeros_like(mask)
    base_mask = np.zeros_like(mask)
    for item in split:
        reconstructed |= item["mask"]
        if not item.get("_is_overlay"):
            base_mask |= item["mask"]
    assert np.array_equal(reconstructed, mask)
    overlay_mask = np.zeros_like(mask)
    for overlay in overlays:
        overlay_mask |= overlay["mask"]
    assert not (overlay_mask & base_mask).any()


def test_component_detection_preserves_tiny_residual_ink():
    animator = make_animator()
    image = np.full((100, 180, 3), 255, dtype=np.uint8)
    image[40:45, 20:150] = 20
    image[10, 170] = 20

    components = animator._find_components(image)

    covered = np.zeros(image.shape[:2], dtype=bool)
    for comp in components:
        covered |= comp["mask"]
    source_ink = np.any(image < 240, axis=2)
    assert np.array_equal(covered, source_ink)
    assert any(comp.get("_is_residual") for comp in components)


def test_single_pixel_residual_has_a_finite_trace_time():
    import numpy as np

    animator = make_animator()
    mask = np.zeros((20, 20), dtype=bool)
    mask[7, 9] = True
    comp = animator._build_component(mask, 1)

    timing = animator._trace_component(comp, 2.0, 0.2)

    assert timing[7, 9] == 2.0


def test_simple_timing_uses_one_directional_front():
    animator = make_animator()
    comp = make_component(100, 100, 500, 200)

    timing = animator._simple_timing(comp["mask"], (150, 300), 0.0, 1.0)

    assert timing[150, 110] < timing[150, 300] < timing[150, 490]


def test_v_stroke_starts_at_endpoint_not_middle_apex():
    import cv2

    animator = make_animator()
    mask = np.zeros((220, 220), dtype=np.uint8)
    cv2.line(mask, (20, 180), (110, 20), 255, 5)
    cv2.line(mask, (110, 20), (200, 180), 255, 5)
    comp = animator._build_component(mask > 0, int((mask > 0).sum()))

    timing = animator._trace_stroke(comp, 0.0, 1.0)
    ys, xs = np.where(comp["mask"])
    first = int(np.argmin(timing[ys, xs]))
    first_point = np.array([ys[first], xs[first]])
    endpoints = [np.array([180, 20]), np.array([180, 200])]
    apex = np.array([20, 110])

    assert min(np.linalg.norm(first_point - point) for point in endpoints) < 15
    assert np.linalg.norm(first_point - apex) > 80


def test_branched_line_art_serializes_paths_instead_of_merging_them():
    animator = make_animator()
    skeleton = np.zeros((40, 80), dtype=bool)
    paths = []
    for index, y in enumerate((10, 20)):
        coords = np.column_stack((
            np.full(30, y, dtype=np.int32),
            np.arange(10, 40, dtype=np.int32),
        ))
        skeleton[coords[:, 0], coords[:, 1]] = True
        paths.append({
            "coords": coords,
            "core": coords,
            "start_node": index * 2 + 1,
            "end_node": index * 2 + 2,
            "start": (float(y), 10.0),
            "end": (float(y), 39.0),
            "top": y,
            "left": 10,
            "start_is_endpoint": True,
            "end_is_endpoint": True,
        })

    strokes = animator._build_line_art_strokes(paths, skeleton)

    assert len(strokes) == 2
    assert all(len(stroke["coords"]) == 30 for stroke in strokes)


def test_junction_paths_continue_straight_instead_of_pinwheeling():
    animator = make_animator()
    skeleton = np.zeros((11, 11), dtype=bool)

    def path(coords, start_node, end_node, *, start_endpoint=False, end_endpoint=False):
        coords = np.asarray(coords, dtype=np.int32)
        skeleton[coords[:, 0], coords[:, 1]] = True
        return {
            "coords": coords,
            "core": coords,
            "start_node": start_node,
            "end_node": end_node,
            "start": tuple(coords[0].astype(float)),
            "end": tuple(coords[-1].astype(float)),
            "top": int(coords[:, 0].min()),
            "left": int(coords[:, 1].min()),
            "start_is_endpoint": start_endpoint,
            "end_is_endpoint": end_endpoint,
        }

    paths = [
        path([(y, 5) for y in range(0, 5)], 1, 10, start_endpoint=True),
        path([(y, 5) for y in range(6, 11)], 10, 2, end_endpoint=True),
        path([(5, x) for x in range(0, 5)], 3, 10, start_endpoint=True),
        path([(5, x) for x in range(6, 11)], 10, 4, end_endpoint=True),
    ]

    strokes = animator._build_line_art_strokes(paths, skeleton)

    endpoint_pairs = {
        frozenset((tuple(stroke["coords"][0]), tuple(stroke["coords"][-1])))
        for stroke in strokes
    }
    assert endpoint_pairs == {
        frozenset(((0, 5), (10, 5))),
        frozenset(((5, 0), (5, 10))),
    }


def test_small_text_junction_uses_path_tracing_instead_of_bfs_fanout():
    animator = make_animator()
    skeleton = np.zeros((9, 9), dtype=bool)
    skeleton[1:8, 4] = True
    skeleton[4, 1:8] = True
    comp = {
        "area": int(skeleton.sum()),
        "is_fill": False,
        "is_text": True,
    }

    assert animator._is_dense_line_art(comp, skeleton)


def test_closed_path_keeps_a_stable_top_left_start():
    animator = make_animator()
    coords = np.asarray([
        (5, 8), (8, 5), (5, 2), (2, 5),
    ], dtype=np.int32)
    skeleton = np.zeros((11, 11), dtype=bool)
    skeleton[coords[:, 0], coords[:, 1]] = True
    paths = [{
        "coords": coords,
        "core": coords,
        "start": tuple(coords[0].astype(float)),
        "end": tuple(coords[-1].astype(float)),
        "top": 2,
        "left": 2,
    }]

    strokes = animator._build_line_art_strokes(paths, skeleton)

    assert tuple(strokes[0]["coords"][0]) == (2, 5)


def test_narrow_filled_marks_are_traced_as_marker_strokes():
    import cv2

    animator = make_animator()
    line_mask = np.zeros((120, 520), dtype=np.uint8)
    cv2.line(line_mask, (20, 60), (500, 60), 255, 28)
    line = animator._build_component(
        line_mask > 0, int((line_mask > 0).sum())
    )

    marker_mask = np.zeros((220, 220), dtype=np.uint8)
    cv2.line(marker_mask, (20, 20), (200, 200), 255, 24)
    cv2.line(marker_mask, (200, 20), (20, 200), 255, 24)
    marker = animator._build_component(
        marker_mask > 0, int((marker_mask > 0).sum())
    )

    block_mask = np.zeros((220, 220), dtype=np.uint8)
    cv2.rectangle(block_mask, (30, 50), (190, 170), 255, -1)
    block = animator._build_component(block_mask > 0, int((block_mask > 0).sum()))

    assert line["is_fill"]
    assert animator._is_marker_stroke(line)
    assert animator._is_marker_stroke(marker)
    assert not animator._is_marker_stroke(block)

    timing = animator._trace_component(line, 0.0, 1.0)
    assert timing[60, 30] < timing[60, 260] < timing[60, 490]


def test_divided_frame_uses_junction_paths_instead_of_contour_bleed():
    import cv2

    animator = make_animator()
    mask = np.zeros((300, 600), dtype=np.uint8)
    cv2.rectangle(mask, (20, 20), (580, 280), 255, 5)
    cv2.line(mask, (210, 20), (210, 280), 255, 5)
    cv2.line(mask, (400, 20), (400, 280), 255, 5)
    comp = animator._build_component(mask > 0, int((mask > 0).sum()))

    def reject_contour(*_args, **_kwargs):
        raise AssertionError("junctioned frame must not use external-contour timing")

    animator._contour_timing = reject_contour
    timing = animator._trace_stroke(comp, 0.0, 1.0)

    assert np.isfinite(timing[comp["mask"]]).all()
    assert float(np.ptp(timing[comp["mask"]])) > 0.8


def test_plain_closed_frame_still_uses_one_contour_front():
    import cv2

    animator = make_animator()
    mask = np.zeros((220, 320), dtype=np.uint8)
    cv2.rectangle(mask, (20, 20), (300, 200), 255, 5)
    comp = animator._build_component(mask > 0, int((mask > 0).sum()))
    calls = []
    original = animator._contour_timing

    def record_contour(*args, **kwargs):
        calls.append(True)
        return original(*args, **kwargs)

    animator._contour_timing = record_contour
    timing = animator._trace_stroke(comp, 0.0, 1.0)

    assert calls
    assert np.isfinite(timing[comp["mask"]]).all()


def test_question_mark_dot_attaches_right_after_its_glyph():
    animator = make_animator()
    # Glyph at top, unrelated text line below, dot of the "?" between them.
    glyph = make_component(500, 100, 540, 200)          # tall question-mark body
    dot = make_component(512, 210, 526, 224)            # small dot just below it
    later_line = make_component(100, 300, 700, 360)     # other content below

    ordered = animator._attach_trailing_dots([glyph, later_line, dot])

    assert ordered == [glyph, dot, later_line]


def test_attach_dots_leaves_unrelated_small_components_alone():
    animator = make_animator()
    glyph = make_component(500, 100, 540, 200)
    far_dot = make_component(50, 600, 64, 614)          # not under the glyph

    ordered = animator._attach_trailing_dots([glyph, far_dot])

    assert ordered == [glyph, far_dot]


def test_element_without_components_is_skipped():
    animator = make_animator()
    shape = (1000, 1000)
    comp = make_component(100, 100, 200, 200)
    plan = [
        element_entry([norm_box(0, 0, 400, 400)], 0.0, 3.0),
        element_entry([norm_box(600, 600, 900, 900)], 3.0, 6.0),
    ]

    schedule = animator._schedule_elements([comp], plan, shape)

    assert len(schedule) == 1
    assert schedule[0][0] is comp
