import datetime

from generate_html_benchmarks import timedelta_format


def test_timedelta_format():
    assert "76:12:30.00000" == timedelta_format(datetime.timedelta(days=3, hours=4, minutes=12, seconds=30))
    assert "04:05:06.00000" == timedelta_format(datetime.timedelta(hours=4, minutes=5, seconds=6))
    assert "11:12:13.00000" == timedelta_format(datetime.timedelta(hours=11, minutes=12, seconds=13))
    assert "00:12:30.00000" == timedelta_format(datetime.timedelta(minutes=12, seconds=30))

