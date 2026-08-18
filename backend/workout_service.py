from workout.workout_service import calculate_angle, normalize_exercise, workout_service


workout_state = {
    "reps": workout_service.reps,
    "stage": workout_service.stage,
}


def process_workout_frame(frame):
    reps = workout_service.process_frame(frame)
    workout_state["reps"] = workout_service.reps
    workout_state["stage"] = workout_service.stage
    return reps
