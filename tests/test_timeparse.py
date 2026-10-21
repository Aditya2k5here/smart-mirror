from datetime import datetime

from core.timeparse import (
    format_duration, humanize_when, normalize_numbers, parse_duration, parse_time_of_day, parse_when,
)

NOW = datetime(2026, 10, 1, 14, 30)  # Thursday


def test_normalize_numbers():
    assert normalize_numbers("twenty five minutes") == "25 minutes"
    assert normalize_numbers("I weigh eighty two") == "i weigh 82"


def test_durations():
    assert parse_duration("10 minutes") == 600
    assert parse_duration("an hour and a half") == 5400
    assert parse_duration("1 hour 30 minutes") == 5400
    assert parse_duration("half an hour") == 1800
    assert parse_duration("ninety seconds") == 90
    assert parse_duration("two and a half hours") == 9000
    assert parse_duration("nothing here") is None


def test_format_duration():
    assert format_duration(90) == "1 minute and 30 seconds" or format_duration(90).startswith("1 minute")
    assert format_duration(3600) == "1 hour"
    assert format_duration(45) == "45 seconds"


def test_time_of_day():
    assert parse_time_of_day("at 6pm") == (18, 0)
    assert parse_time_of_day("6:30 am") == (6, 30)
    assert parse_time_of_day("at 18:45") == (18, 45)
    assert parse_time_of_day("noon") == (12, 0)
    # 2:30pm now -> "at 6" means 6pm
    assert parse_time_of_day("at 6", NOW) == (18, 0)


def test_when_relative():
    when, rest = parse_when("call mom in twenty minutes", NOW)
    assert when == datetime(2026, 10, 1, 14, 50)
    assert rest == "call mom"


def test_when_clock_and_tomorrow():
    when, rest = parse_when("buy milk tomorrow at 9am", NOW)
    assert when == datetime(2026, 10, 2, 9, 0)
    assert rest == "buy milk"
    when, _ = parse_when("stretch at 6pm", NOW)
    assert when == datetime(2026, 10, 1, 18, 0)
    when, _ = parse_when("wake up at 7am", NOW)
    assert when == datetime(2026, 10, 2, 7, 0)  # already past today


def test_when_weekday_and_dates():
    when, _ = parse_when("gym on friday at 6am", NOW)
    assert when == datetime(2026, 10, 2, 6, 0)
    when, _ = parse_when("by december 31", NOW)
    assert (when.month, when.day) == (12, 31)
    when, _ = parse_when("in 3 months", NOW)
    assert (when.year, when.month) == (2027, 1)
    when, _ = parse_when("by 15th of march", NOW)
    assert (when.year, when.month, when.day) == (2027, 3, 15)


def test_when_none():
    when, rest = parse_when("buy some milk", NOW)
    assert when is None and rest == "buy some milk"


def test_humanize():
    assert humanize_when(datetime(2026, 10, 1, 14, 50), NOW) == "in 20 minutes"
    assert humanize_when(datetime(2026, 10, 2, 9, 0), NOW) == "tomorrow at 9 AM"
