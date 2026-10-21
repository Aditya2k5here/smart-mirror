from workout.exercises import RepCounter, angle, find_exercise


def lm(**points):
    return {name: (x, y, 1.0) for name, (x, y) in points.items()}


def arm(elbow_angle_deg):
    # shoulder above elbow, wrist rotated around elbow by the given interior angle
    import math
    ex, ey = 0.5, 0.5
    sx, sy = 0.5, 0.3
    rad = math.radians(180 - elbow_angle_deg)
    wx, wy = ex + 0.2 * math.sin(rad), ey + 0.2 * math.cos(rad)
    return lm(right_shoulder=(sx, sy), right_elbow=(ex, ey), right_wrist=(wx, wy))


def test_find_exercise():
    assert find_exercise("start push ups") == "pushups"
    assert find_exercise("let's do shoulder press") == "shoulder_press"
    assert find_exercise("jumping jacks please") == "jumping_jacks"
    assert find_exercise("anything") == "curls"


def test_angle():
    assert round(angle((0, 1), (0, 0), (1, 0))) == 90
    assert round(angle((0, 1), (0, 0), (0, 2))) == 0


def test_curl_counts_once_per_rep():
    c = RepCounter("curls")
    for deg in (170, 120, 60, 30, 60, 120, 170, 90, 30, 170):
        c.update(arm(deg))
    assert c.reps == 2


def test_curl_needs_full_extension_first():
    c = RepCounter("curls")
    for deg in (100, 30, 100, 30):  # never extends past 155
        c.update(arm(deg))
    assert c.reps == 0


def test_squats_and_lunges():
    def legs(deg_left, deg_right):
        import math
        out = {}
        for side, deg in (("left", deg_left), ("right", deg_right)):
            kx, ky = 0.5, 0.6
            out[f"{side}_hip"] = (kx, ky - 0.2, 1.0)
            rad = math.radians(180 - deg)
            out[f"{side}_knee"] = (kx, ky, 1.0)
            out[f"{side}_ankle"] = (kx + 0.2 * math.sin(rad), ky + 0.2 * math.cos(rad), 1.0)
        return out

    s = RepCounter("squats")
    for d in (170, 130, 90, 130, 170, 90, 170):
        s.update(legs(d, d))
    assert s.reps == 2

    l = RepCounter("lunges")
    for dl, dr in ((170, 170), (90, 140), (170, 170), (140, 90), (170, 170)):
        l.update(legs(dl, dr))
    assert l.reps == 2


def test_jumping_jacks():
    c = RepCounter("jumping_jacks")
    closed = lm(left_wrist=(0.45, 0.6), right_wrist=(0.55, 0.6), left_ankle=(0.47, 0.9), right_ankle=(0.53, 0.9),
                left_shoulder=(0.4, 0.4), right_shoulder=(0.6, 0.4), nose=(0.5, 0.25))
    opened = lm(left_wrist=(0.3, 0.1), right_wrist=(0.7, 0.1), left_ankle=(0.25, 0.9), right_ankle=(0.75, 0.9),
                left_shoulder=(0.4, 0.4), right_shoulder=(0.6, 0.4), nose=(0.5, 0.25))
    for frame in (closed, opened, closed, opened, closed):
        c.update(frame)
    assert c.reps == 2


def test_plank_hold_time():
    t = [0.0]
    c = RepCounter("plank", clock=lambda: t[0])
    flat = lm(left_shoulder=(0.2, 0.5), left_hip=(0.5, 0.5), left_ankle=(0.8, 0.5))
    sag = lm(left_shoulder=(0.2, 0.5), left_hip=(0.5, 0.65), left_ankle=(0.8, 0.5))
    c.update(flat)
    t[0] = 10
    c.update(flat)
    assert round(c.held) == 10
    c.update(sag)
    t[0] = 20
    assert round(c.held) == 10  # hold stopped when form broke
