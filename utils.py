from contextlib import contextmanager
from datetime import datetime, timedelta, date, timezone
import inspect
import time
from typing import Any
import streamlit as st

DATE_TYPE_DAY, DATE_TYPE_MONTH, DATE_TYPE_YEAR, DATE_TYPE_WEEK = 0, 1, 2, 3
PRIVACY_MODE: bool = False


@contextmanager
def time_it(msg: str) -> Any:
    start = time.time()
    try:
        yield start
    finally:
        print(f"{msg} – {int((time.time() - start)*1000)}ms")


def timeit(func):
    async def timed_async(*args, **kwargs):
        start = time.time()
        try:
            return await func(*args, **kwargs)
        finally:
            print(f"{func.__name__} – {int((time.time() - start)*1000)}ms")

    def timed_sync(*args, **kwargs):
        start = time.time()
        try:
            return func(*args, **kwargs)
        finally:
            print(f"{func.__name__} – {int((time.time() - start)*1000)}ms")
    return timed_async if inspect.iscoroutinefunction(func) else timed_sync


def _this(date_type: int) -> date:
    dt = datetime.now().date()
    if date_type == DATE_TYPE_DAY:
        return dt
    if date_type == DATE_TYPE_MONTH:
        return dt.replace(day=1)
    if date_type == DATE_TYPE_YEAR:
        return dt.replace(day=1, month=1)
    if date_type == DATE_TYPE_WEEK:
        return dt - timedelta(days=dt.weekday())
    raise ValueError(date_type)


def _last(date_type: str) -> date:
    dt = datetime.now().date()
    if date_type == DATE_TYPE_DAY:
        return dt - timedelta(days=1)
    if date_type == DATE_TYPE_MONTH:
        return (dt.replace(day=1) - timedelta(days=1)).replace(day=1)
    if date_type == DATE_TYPE_YEAR:
        return dt.replace(day=1, month=1, year=dt.year-1)
    if date_type == DATE_TYPE_WEEK:
        return dt - timedelta(days=dt.weekday()+7)
    raise ValueError(date_type)


def make_utc(dt: date) -> datetime:
    timestamp = time.mktime(dt.timetuple())
    return datetime.fromtimestamp(timestamp, tz=timezone.utc)


class Colors:
    RED = "#C06060"
    GREEN = "#468B52"
    ORANGE = "#A17435"
    GREY = "#999999"
    WHITE = "#EAEAEA"
    BLUE = "#1F82C5"


def write_span(inner: str, color: str = "", size: str = "", style: str = "", title: str = "", outer_div: bool = False, div_style: str = ""):
    write_html(render_span(inner, color, size, style, title, outer_div, div_style))


def write_html(html: str):
    st.write(html, unsafe_allow_html=True)


def render_span(inner: str, color: str = "", size: str = "", style: str = "", title: str = "", outer_div: bool = False, div_style: str = "") -> str:
    color = f'color:{color};' if color else ""
    size = f'font-size:{size};' if size else ""
    title = f' title="{title}"' if title else ""
    html = f'<span style="{color}{size}{style}"{title}>{inner}</span>'
    div_style = f' style="{div_style}"' if div_style else ""
    return f'<div{div_style}>{html}</div>' if outer_div else html


def render_date_title(option: int | None = None, prefix: str = "In"):
    option = st.session_state.get("date_type", 0) if option is None else option
    date = st.session_state.get("date_picker") or _this(DATE_TYPE_DAY)
    if option == DATE_TYPE_DAY:
        if date == _this(DATE_TYPE_DAY):
            return "Today"
        elif date == _last(DATE_TYPE_DAY):
            return "Yesterday"
        add_year = "" if date.year == _this(DATE_TYPE_DAY).year else f"'{date:%y}"
        return f'{date:%a, %d %B}{add_year}'
    if option == DATE_TYPE_MONTH:
        add_year = "" if date.year == _this(DATE_TYPE_MONTH).year else f"'{date:%y}"
        return f'{prefix} {date:%B}{add_year}'
    elif option == DATE_TYPE_YEAR:
        return f'{prefix} {date.year}'
    elif option == DATE_TYPE_WEEK:
        week_start = date - timedelta(days=date.weekday())
        return f'{week_start:%d %b} – {week_start + timedelta(days=6):%d %b}'
    else:
        raise ValueError(option)


def write_date_picker(columns: list | None = None) -> tuple[date, date] | None:
    def change_date_type():
        st.session_state["date_type"] = st.session_state.get("date_type")

    def change_date(option: int, direction: int):
        date = st.session_state.get("date_picker") or _this(DATE_TYPE_DAY)
        if option == DATE_TYPE_YEAR:
            st.session_state["date_picker"] = date.replace(year=date.year + direction)
        elif option == DATE_TYPE_DAY:
            st.session_state["date_picker"] = date + timedelta(days=direction)
        elif option == DATE_TYPE_MONTH:
            st.session_state["date_picker"] = (date.replace(day=1) + timedelta(days=(31 if direction > 0 else -1))).replace(day=1)
        elif option == DATE_TYPE_WEEK:
            st.session_state["date_picker"] = date + timedelta(days=-date.weekday() + 7 * direction)

        st.session_state["date_type"] = st.session_state.get("date_type")

    col1, col2, col3, *_ = columns if columns else st.columns((27.5, 75, 22.5, 500))
    date = st.session_state.get("date_picker") or _this(DATE_TYPE_DAY)
    with col2:
        option = st.radio("Period", [DATE_TYPE_DAY, DATE_TYPE_WEEK, DATE_TYPE_MONTH, DATE_TYPE_YEAR],
                          format_func=render_date_title, key="date_type", on_change=change_date_type)

    with col1:
        write_span("&nbsp;", outer_div=True, div_style="margin-top:-5px")
        st.button("◀", on_click=change_date, args=(option, -1))

    with col3:
        write_span("&nbsp;", outer_div=True, div_style="margin-top:-5px")
        if (((option == DATE_TYPE_DAY) and (date < _this(DATE_TYPE_DAY))) or
            ((option == DATE_TYPE_WEEK) and (date - timedelta(days=date.weekday()) < _this(DATE_TYPE_WEEK))) or
            ((option == DATE_TYPE_MONTH) and (date.replace(day=1) < _this(DATE_TYPE_MONTH))) or
                ((option == DATE_TYPE_YEAR) and (date.replace(month=1) < _this(DATE_TYPE_YEAR)))):
            st.button("▶", on_click=change_date, args=(option, 1))

    return ((date, (date + timedelta(days=1))) if option == DATE_TYPE_DAY else
            ((start := date.replace(day=1)), (start + timedelta(days=31)).replace(day=1)) if option == DATE_TYPE_MONTH else
            ((start := date.replace(month=1, day=1)), start.replace(year=date.year+1)) if option == DATE_TYPE_YEAR else
            ((start := date - timedelta(days=date.weekday())), (start + timedelta(days=7))) if option == DATE_TYPE_WEEK else
            None)
